# GPU kernels for the lean JSON parsing pipeline.
#
# `fused_json_kernel` is the only kernel `gpu/parser.mojo` launches.
# It walks 32 input bytes per thread and emits two 32-bit-per-thread
# bitmaps:
#
#   * `output_structural` -- bytes that are one of `{ } [ ] : ,`
#     (the raw mask -- the kernel intentionally does not filter
#     in-string occurrences; correctness lives in
#     `gpu/tape_adapter._result_to_index` on the CPU side).
#   * `output_open_close` -- bytes that are one of `{ } [ ]` only.
#
# `quote_prefix_in` is still in the signature (read once and discarded)
# so a future, correct GPU-side escape implementation can drop in here
# without changing the host launch site.
#
# `popcount_fast` is also exported because `stream_compact.mojo`
# (and downstream `extract_positions_gpu_lean`) reuses it for the
# 32-bit-per-word popcount step of the GPU stream compaction.

from max.gpu import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.globals import MAX_THREADS_PER_BLOCK_METADATA
from std.memory import UnsafePointer
from std.utils.static_tuple import StaticTuple
from ..types import (
    CHAR_OPEN_BRACE,
    CHAR_CLOSE_BRACE,
    CHAR_OPEN_BRACKET,
    CHAR_CLOSE_BRACKET,
    CHAR_COLON,
    CHAR_COMMA,
    CHAR_QUOTE,
    CHAR_BACKSLASH,
    CHAR_NEWLINE,
)


# Block size for GPU kernels (256 threads typical for good occupancy)
comptime BLOCK_SIZE_OPT: Int = 256


@always_inline
def popcount_fast(value: UInt32) -> UInt32:
    """Fast popcount using hardware instruction if available."""
    var v = value
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    v = v * 0x01010101
    return v >> 24


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(Int(BLOCK_SIZE_OPT))
    )
)
def fused_json_kernel(
    input_data: UnsafePointer[UInt8, MutAnyOrigin],
    output_structural: UnsafePointer[UInt32, MutAnyOrigin],
    output_open_close: UnsafePointer[UInt32, MutAnyOrigin],
    quote_prefix_in: UnsafePointer[UInt32, MutAnyOrigin],
    size: UInt32,
    total_padded_32: UInt32,
):
    """Walk 32 input bytes per thread; emit raw `{}[]:,` and `{}[]` bitmaps.

    Each thread processes 32 bytes -> produces one 32-bit bitmap word per
    output stream.

    NOTE: in-string detection on the GPU side is intentionally NOT done
    here. The previous formulation

        escaped       = quote_bits & (slash_bits << 1)
        real_quotes   = quote_bits & ~escaped
        pxor          = prefix_xor_fast(real_quotes)
        in_string     = pxor (xor-flipped by quote_prefix_in[global_id]&1)
        structural    = (~in_string) & op_bits

    is *not* a correct implementation of the simdjson escape model.
    `slash_bits << 1` only catches `\\\"` *within* a 32-byte chunk, so
    a backslash at byte 31 of one chunk followed by a quote at byte
    0 of the next chunk is not caught. It also doesn't distinguish
    odd vs even backslash runs, so `\\\\\"` (literal `\\` followed by a
    real quote) is wrongly classified as an escaped quote.

    The fix is to emit the *raw* `{}[]:,` bitmap and have the CPU
    side (`tape_adapter._result_to_index`) drop positions that fall
    inside string literals using its own correct, byte-by-byte
    escape state machine. The CPU pass was already walking the input
    to recover quote positions for stage 2, so this adds no extra
    scan -- it just uses the existing walk to filter the GPU output.

    `quote_prefix_in` stays in the signature so a future, correct
    GPU escape implementation can drop in here without changing the
    call site.
    """
    var thread_id = Int(thread_idx.x)
    var block_id = Int(block_idx.x)
    var global_id = block_id * Int(block_dim.x) + thread_id

    if global_id >= Int(total_padded_32):
        return

    var start_pos = global_id * 32

    var op_bits: UInt32 = 0
    var open_close_bits: UInt32 = 0

    # Step 1: Build bitmaps by scanning 32 bytes
    # Use manual unrolling for better instruction-level parallelism
    comptime for j in range(32):
        var pos = start_pos + j
        if pos >= Int(size):
            break

        var c = input_data[pos]
        var bit_mask = UInt32(1) << UInt32(j)

        var is_op = (
            (c == CHAR_OPEN_BRACE)
            | (c == CHAR_CLOSE_BRACE)
            | (c == CHAR_OPEN_BRACKET)
            | (c == CHAR_CLOSE_BRACKET)
            | (c == CHAR_COLON)
            | (c == CHAR_COMMA)
        )
        op_bits |= bit_mask * UInt32(is_op)

        var is_bracket = (
            (c == CHAR_OPEN_BRACE)
            | (c == CHAR_CLOSE_BRACE)
            | (c == CHAR_OPEN_BRACKET)
            | (c == CHAR_CLOSE_BRACKET)
        )
        open_close_bits |= bit_mask * UInt32(is_bracket)

    # Read-and-discard the (unused) quote_prefix_in argument so the
    # buffer binding survives Metal AOT alias analysis. See file-top
    # docstring + parser.mojo `d_quote_dummy` comment.
    _ = quote_prefix_in[global_id]

    output_structural[global_id] = op_bits
    output_open_close[global_id] = open_close_bits
