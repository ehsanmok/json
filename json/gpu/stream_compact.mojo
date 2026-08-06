# GPU Stream Compaction for Position Extraction
#
# This module implements GPU-parallel stream compaction to extract
# structural character positions from a bitmap.
#
# Algorithm:
# 1. Popcount: Each thread computes popcount of its 32-bit bitmap word
# 2. Prefix Sum: Exclusive prefix sum of popcounts gives write offsets
# 3. Scatter: Each thread writes positions using CTZ to extract set bits
#
# The lean pipeline only needs positions (the CPU tape adapter does
# not consume a per-position `char_types` companion stream), so the
# `_lean` variant below is the only public scatter path.

from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu import barrier
from max.gpu.primitives import block
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.globals import MAX_THREADS_PER_BLOCK_METADATA
from std.collections import List
from std.memory import Pointer, memcpy
from std.math import ceildiv
from std.utils.static_tuple import StaticTuple

from .kernels import popcount_fast, BLOCK_SIZE_OPT


# ===== Kernel 1: Popcount each bitmap word =====
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(Int(BLOCK_SIZE_OPT))
    )
)
def popcount_kernel(
    bitmap: Pointer[UInt32, MutAnyOrigin],
    popcounts: Pointer[UInt32, MutAnyOrigin],
    num_words: UInt32,
):
    """Compute popcount of each bitmap word."""
    var gid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if gid >= Int(num_words):
        return
    popcounts[gid] = popcount_fast(bitmap[gid])


# ===== Kernel 2: Parallel block-local exclusive prefix sum =====
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(Int(BLOCK_SIZE_OPT))
    )
)
def prefix_sum_kernel(
    input_data: Pointer[UInt32, MutAnyOrigin],
    output_prefix: Pointer[UInt32, MutAnyOrigin],
    block_totals: Pointer[UInt32, MutAnyOrigin],
    num_elements: UInt32,
):
    """Compute a per-block exclusive prefix sum using `block.prefix_sum`.

    Each block scans `BLOCK_SIZE_OPT` elements in parallel and stores its
    grand total into `block_totals[block_id]`. Callers must launch this with
    `block_dim=BLOCK_SIZE_OPT` and a grid large enough to cover `num_elements`;
    the hierarchical helper `_compute_block_prefix_sums` then aggregates block
    totals across levels.
    """
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var gid = bid * Int(block_dim.x) + tid

    var val: UInt32 = 0
    if gid < Int(num_elements):
        val = input_data[gid]

    var prefix = block.prefix_sum[exclusive=True, block_size=BLOCK_SIZE_OPT](
        val
    )

    if gid < Int(num_elements):
        output_prefix[gid] = prefix

    # Last active thread in this block writes the block total.
    var block_end = min((bid + 1) * Int(block_dim.x), Int(num_elements))
    var last_in_block = block_end - 1 - bid * Int(block_dim.x)

    if tid == last_in_block:
        block_totals[bid] = prefix + val


# ===== Kernel 3: Add block offsets to prefix sums =====
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(Int(BLOCK_SIZE_OPT))
    )
)
def add_block_offsets_kernel(
    prefix_sums: Pointer[UInt32, MutAnyOrigin],
    block_offsets: Pointer[UInt32, MutAnyOrigin],
    num_elements: UInt32,
):
    """Add block offset to each element's prefix sum."""
    var tid = Int(thread_idx.x)
    var block_id = Int(block_idx.x)
    var gid = Int(block_dim.x) * block_id + tid

    if gid >= Int(num_elements):
        return

    # Skip first block (offset is 0)
    if block_id > 0:
        prefix_sums[gid] = prefix_sums[gid] + block_offsets[block_id]


def _ctz32_gpu(value: UInt32) -> UInt32:
    """Count trailing zeros (GPU version)."""
    if value == 0:
        return 32

    var n: UInt32 = 0
    var v = value

    if (v & 0x0000FFFF) == 0:
        n += 16
        v >>= 16
    if (v & 0x000000FF) == 0:
        n += 8
        v >>= 8
    if (v & 0x0000000F) == 0:
        n += 4
        v >>= 4
    if (v & 0x00000003) == 0:
        n += 2
        v >>= 2
    if (v & 0x00000001) == 0:
        n += 1

    return n


# ===== Helper: Recursive hierarchical prefix sum =====
def _compute_block_prefix_sums(
    ctx: DeviceContext,
    d_block_totals_ptr: Pointer[UInt32, MutAnyOrigin],
    d_block_prefix_ptr: Pointer[UInt32, MutAnyOrigin],
    num_blocks: Int,
) raises:
    """Recursively compute prefix sum of block totals.

    Handles arbitrary levels of hierarchy.
    """
    if num_blocks <= BLOCK_SIZE_OPT:
        # Base case: single block can handle all totals
        var d_dummy = ctx.enqueue_create_buffer[DType.uint32](1)
        d_dummy.enqueue_fill(0)

        ctx.enqueue_function[prefix_sum_kernel](
            d_block_totals_ptr,
            d_block_prefix_ptr,
            d_dummy.unsafe_ptr(),
            UInt32(num_blocks),
            grid_dim=1,
            block_dim=BLOCK_SIZE_OPT,
        )
        return

    # Multiple blocks needed for this level
    var num_blocks_l1 = ceildiv(num_blocks, BLOCK_SIZE_OPT)

    # Compute block-local prefix sums
    var d_block_totals_l1 = ctx.enqueue_create_buffer[DType.uint32](
        num_blocks_l1
    )
    d_block_totals_l1.enqueue_fill(0)

    ctx.enqueue_function[prefix_sum_kernel](
        d_block_totals_ptr,
        d_block_prefix_ptr,
        d_block_totals_l1.unsafe_ptr(),
        UInt32(num_blocks),
        grid_dim=num_blocks_l1,
        block_dim=BLOCK_SIZE_OPT,
    )

    # Recursively handle the next level
    var d_block_prefix_l1 = ctx.enqueue_create_buffer[DType.uint32](
        num_blocks_l1
    )
    d_block_prefix_l1.enqueue_fill(0)

    _compute_block_prefix_sums(
        ctx,
        d_block_totals_l1.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        d_block_prefix_l1.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        num_blocks_l1,
    )

    # Propagate offsets back down
    ctx.enqueue_function[add_block_offsets_kernel](
        d_block_prefix_ptr,
        d_block_prefix_l1.unsafe_ptr(),
        UInt32(num_blocks),
        grid_dim=num_blocks_l1,
        block_dim=BLOCK_SIZE_OPT,
    )


def extract_positions_gpu_lean(
    ctx: DeviceContext,
    d_bitmap_ptr: Pointer[UInt32, MutAnyOrigin],
    num_words: Int,
    max_byte_pos: Int,
) raises -> List[Int32]:
    """Stream-compact a structural bitmap into a position list.

    Positions-only: the tape adapter
    (`gpu/tape_adapter.parse_gpu_to_value`) only consumes
    `result.structural`, so neither a per-position `char_types`
    companion stream nor a `pair_pos` array is produced here.

    Returns the positions list; caller already knows `max_byte_pos`.
    """
    var num_blocks = ceildiv(num_words, BLOCK_SIZE_OPT)

    var d_popcounts = ctx.enqueue_create_buffer[DType.uint32](num_words)
    ctx.enqueue_function[popcount_kernel](
        d_bitmap_ptr,
        d_popcounts.unsafe_ptr(),
        UInt32(num_words),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE_OPT,
    )

    var d_prefix = ctx.enqueue_create_buffer[DType.uint32](num_words)
    var d_block_totals = ctx.enqueue_create_buffer[DType.uint32](num_blocks)
    d_block_totals.enqueue_fill(0)

    ctx.enqueue_function[prefix_sum_kernel](
        d_popcounts.unsafe_ptr(),
        d_prefix.unsafe_ptr(),
        d_block_totals.unsafe_ptr(),
        UInt32(num_words),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE_OPT,
    )

    var total_count: Int
    if num_blocks == 1:
        ctx.synchronize()
        var h_block_totals = ctx.enqueue_create_host_buffer[DType.uint32](
            num_blocks
        )
        ctx.enqueue_copy(h_block_totals, d_block_totals)
        ctx.synchronize()
        total_count = Int(h_block_totals.unsafe_ptr()[0])
    else:
        var d_block_prefix = ctx.enqueue_create_buffer[DType.uint32](num_blocks)
        d_block_prefix.enqueue_fill(0)

        _compute_block_prefix_sums(
            ctx,
            d_block_totals.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            d_block_prefix.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            num_blocks,
        )

        ctx.enqueue_function[add_block_offsets_kernel](
            d_prefix.unsafe_ptr(),
            d_block_prefix.unsafe_ptr(),
            UInt32(num_words),
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE_OPT,
        )

        ctx.synchronize()
        var h_block_totals = ctx.enqueue_create_host_buffer[DType.uint32](
            num_blocks
        )
        var h_block_prefix = ctx.enqueue_create_host_buffer[DType.uint32](
            num_blocks
        )
        ctx.enqueue_copy(h_block_totals, d_block_totals)
        ctx.enqueue_copy(h_block_prefix, d_block_prefix)
        ctx.synchronize()
        total_count = Int(h_block_prefix.unsafe_ptr()[num_blocks - 1]) + Int(
            h_block_totals.unsafe_ptr()[num_blocks - 1]
        )

    if total_count == 0:
        return List[Int32]()

    var d_positions = ctx.enqueue_create_buffer[DType.int32](total_count)
    d_positions.enqueue_fill(0)

    ctx.enqueue_function[scatter_positions_lean_kernel](
        d_bitmap_ptr,
        d_prefix.unsafe_ptr(),
        d_positions.unsafe_ptr(),
        UInt32(num_words),
        UInt32(max_byte_pos),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE_OPT,
    )

    var h_positions = ctx.enqueue_create_host_buffer[DType.int32](total_count)
    ctx.enqueue_copy(h_positions, d_positions)
    ctx.synchronize()

    var positions = List[Int32](capacity=total_count)
    positions.resize(total_count, 0)
    memcpy(
        dest=positions.unsafe_ptr(),
        src=h_positions.unsafe_ptr(),
        count=total_count,
    )

    return positions^


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(Int(BLOCK_SIZE_OPT))
    )
)
def scatter_positions_lean_kernel(
    bitmap: Pointer[UInt32, MutAnyOrigin],
    prefix_offsets: Pointer[UInt32, MutAnyOrigin],
    output_positions: Pointer[Int32, MutAnyOrigin],
    num_words: UInt32,
    max_byte_pos: UInt32,
):
    """Scatter set-bit byte positions out of a 32-bit-per-word bitmap.

    Each thread owns one 32-bit word; CTZ extracts set-bit indices in
    ascending order and writes them at `prefix_offsets[gid]`. No
    input byte load and no companion byte-type output buffer -- the
    tape adapter recomputes character classes from the input bytes.
    """
    var gid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if gid >= Int(num_words):
        return

    var bits = bitmap[gid]
    if bits == 0:
        return

    var base_pos = gid * 32
    var write_idx = Int(prefix_offsets[gid])

    while bits != 0:
        var tz = _ctz32_gpu(bits)
        var pos = base_pos + Int(tz)

        if pos < Int(max_byte_pos):
            output_positions[write_idx] = Int32(pos)
            write_idx += 1

        bits = bits & (bits - 1)
