# GPU JSON parser entry points -- unified lean pipeline.
#
# Both `parse_json_gpu` (owns its input) and
# `parse_json_gpu_from_pinned` (caller hands us an already-pinned host
# buffer) funnel into the same `_parse_lean` core. The core launches a
# single `fused_json_kernel`, runs positions-only stream compaction
# (`extract_positions_gpu_lean`), and returns a `JSONResult` whose
# `pair_pos` is a -1-filled placeholder kept for legacy-iterator ABI
# compatibility.
#
# Why "lean"
# ----------
# `gpu/kernels.mojo:fused_json_kernel` intentionally emits the raw
# `{}[]:,` bitmap and ignores the `quote_prefix_in` argument (the
# correct, byte-by-byte escape state machine lives in
# `gpu/tape_adapter._result_to_index` on the CPU side). That makes the
# popcount + hierarchical prefix-sum cascade that used to feed
# `quote_prefix_in` pure dead work; the CPU `_match_brackets_fast` pass
# that built `pair_pos` is also dead because `parse_gpu_to_value` only
# consumes `gpu_result.structural`. The Apple Metal path proved this
# out; this revision makes the same pipeline canonical on NVIDIA / AMD.
#
# Per-input footprint drops from ~9 device buffers to 4 and the wall
# clock loses the CPU bracket-match phase (~100+ ms on
# twitter_large_record on B200). On Apple unified-memory hosts where
# 800 MB inputs used to crash the host, the chunked variant still lives
# behind `JSON_GPU_APPLE_CHUNK_MB` (default 64 MB; opt-out via
# `JSON_GPU_APPLE_NO_CHUNK=1` for single-shot A/B).
#
# Note: `fused_json_kernel`'s signature still takes a `quote_prefix_in`
# pointer (kept stable so a correct GPU-side escape implementation can
# drop in later). We pass a dedicated `d_quote_dummy` device buffer that
# nothing reads for correctness -- Metal AOT rejects aliasing with a
# write target, so it cannot be reused for any of the output buffers.

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu import block_dim, block_idx, thread_idx, barrier, global_idx
from std.gpu.globals import MAX_THREADS_PER_BLOCK_METADATA
from std.collections import List
from std.memory import UnsafePointer, memcpy
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.utils.static_tuple import StaticTuple

from ..types import JSONInput, JSONResult
from .kernels import BLOCK_SIZE_OPT, fused_json_kernel
from .stream_compact import extract_positions_gpu_lean


def parse_json_gpu(
    var input: JSONInput, verbose: Bool = False
) raises -> JSONResult:
    """Parse a JSON document on the GPU. Allocates its own
    DeviceContext + pinned host buffer.

    Args:
        input: Owned byte buffer with the JSON document.
        verbose: If True, print per-phase timings to stdout. Useful for
            `pixi run bench-gpu -- --debug-timing <file>`.

    Returns:
        `JSONResult` whose `structural` carries the in-order
        `{ } [ ] : ,` byte offsets (the GPU side does not classify
        in-string bytes; `gpu/tape_adapter` does that on the CPU).
        `pair_pos` is a -1-filled placeholder.
    """
    comptime if not has_accelerator():
        raise Error(
            "parse_json_gpu requires a supported accelerator at compile"
            " time. None was detected on this host (has_accelerator()"
            " returned False). Use loads(...) without target='gpu' to run"
            " on the CPU backend."
        )

    var size = len(input.data)
    if size == 0:
        var result = JSONResult()
        return result^

    var total_padded_32 = (size + 31) // 32
    var ctx = DeviceContext()

    var t0 = perf_counter_ns()
    var d_input = ctx.enqueue_create_buffer[DType.uint8](size)
    var h_input = ctx.enqueue_create_host_buffer[DType.uint8](size)
    memcpy(dest=h_input.unsafe_ptr(), src=input.data.unsafe_ptr(), count=size)
    ctx.enqueue_copy(d_input, h_input)

    return _parse_lean(ctx, d_input, size, total_padded_32, t0, verbose)


def parse_json_gpu_from_pinned(
    ctx: DeviceContext,
    h_input: HostBuffer[DType.uint8],
    size: Int,
    verbose: Bool = False,
) raises -> JSONResult:
    """Parse a JSON document on the GPU from a caller-owned pinned host
    buffer. Skips the host->pinned memcpy + DeviceContext setup, which
    is the steady-state hot path (e.g. the bench harness).

    Args:
        ctx: Device context (reused across calls in production).
        h_input: Pinned host buffer with the JSON document already
            copied in.
        size: Length of the document in bytes.
        verbose: If True, print per-phase timings to stdout.
    """
    comptime if not has_accelerator():
        raise Error(
            "parse_json_gpu_from_pinned requires a supported accelerator at"
            " compile time. None was detected on this host"
            " (has_accelerator() returned False)."
        )

    if size == 0:
        var result = JSONResult()
        return result^

    var total_padded_32 = (size + 31) // 32

    var t0 = perf_counter_ns()
    var d_input = ctx.enqueue_create_buffer[DType.uint8](size)
    ctx.enqueue_copy(d_input, h_input)

    return _parse_lean(ctx, d_input, size, total_padded_32, t0, verbose)


def _parse_lean(
    ctx: DeviceContext,
    d_input: DeviceBuffer[DType.uint8],
    size: Int,
    total_padded_32: Int,
    t0: UInt,
    verbose: Bool,
) raises -> JSONResult:
    """Lean kernel + extract. Single fused kernel launch (raw
    `{}[]:,` bitmap), positions-only stream compaction, no GPU
    in-string mask, no CPU bracket-match pass. Used by every backend.
    """
    var result = JSONResult()
    result.file_size = size

    var num_blocks = ceildiv(total_padded_32, BLOCK_SIZE_OPT)
    if num_blocks == 0:
        num_blocks = 1

    var d_structural = ctx.enqueue_create_buffer[DType.uint32](total_padded_32)
    var d_open_close = ctx.enqueue_create_buffer[DType.uint32](total_padded_32)
    # Read-only dummy for `fused_json_kernel`'s `quote_prefix_in`
    # parameter (the kernel reads it once and discards; Metal AOT
    # rejects aliasing with a write target so it must be its own
    # buffer).
    var d_quote_dummy = ctx.enqueue_create_buffer[DType.uint32](total_padded_32)

    if verbose:
        ctx.synchronize()
        var t_h2d = perf_counter_ns()
        print("    H2D + alloc:", Float64(t_h2d - t0) / 1e6, "ms")

    ctx.enqueue_function[fused_json_kernel](
        d_input.unsafe_ptr(),
        d_structural.unsafe_ptr(),
        d_open_close.unsafe_ptr(),
        d_quote_dummy.unsafe_ptr(),
        UInt(size),
        UInt(total_padded_32),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE_OPT,
    )

    ctx.synchronize()
    var t1 = perf_counter_ns()
    if verbose:
        print("    fused kernel:", Float64(t1 - t0) / 1e6, "ms (cumulative)")

    # Positions-only stream compaction: skips the char_types scatter
    # + D2H copy (~66 MB on twitter_large_record) since
    # `_match_brackets_fast` is gone.
    result.structural = extract_positions_gpu_lean(
        ctx,
        d_structural.unsafe_ptr(),
        total_padded_32,
        size,
    )
    # `pair_pos` is unused by `gpu/tape_adapter.parse_gpu_to_value`,
    # but downstream legacy-iterator code still indexes it. A
    # -1-filled list sized to `structural` keeps that ABI without
    # paying for bracket matching.
    result.pair_pos = List[Int32](capacity=len(result.structural))
    result.pair_pos.resize(len(result.structural), -1)

    if verbose:
        var t2 = perf_counter_ns()
        var total_ms = Float64(t2 - t0) / 1e6
        var gbps = (Float64(size) / 1e9) / (total_ms / 1e3)
        print(
            "    position extract:",
            Float64(t2 - t1) / 1e6,
            "ms; total",
            total_ms,
            "ms,",
            gbps,
            "GB/s,",
            len(result.structural),
            "structurals",
        )

    return result^
