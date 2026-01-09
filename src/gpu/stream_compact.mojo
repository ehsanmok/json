# GPU Stream Compaction for Position Extraction
#
# This module implements GPU-parallel stream compaction to extract
# structural character positions from a bitmap.
#
# Algorithm:
# 1. Popcount: Each thread computes popcount of its 32-bit bitmap word
# 2. Prefix Sum: Exclusive prefix sum of popcounts gives write offsets
# 3. Scatter: Each thread writes positions using CTZ to extract set bits

from gpu.host import DeviceContext, DeviceBuffer
from gpu import block_dim, block_idx, thread_idx, barrier, block
from collections import List
from memory import UnsafePointer, memcpy
from math import ceildiv

from .kernels import popcount_fast, BLOCK_SIZE_OPT


# ===== Kernel 1: Popcount each bitmap word =====
fn popcount_kernel(
    bitmap: UnsafePointer[UInt32, MutAnyOrigin],
    popcounts: UnsafePointer[UInt32, MutAnyOrigin],
    num_words: UInt,
):
    """Compute popcount of each bitmap word."""
    var gid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if gid >= Int(num_words):
        return
    popcounts[gid] = popcount_fast(bitmap[gid])


# ===== Kernel 2: Block-level prefix sum with block totals =====
fn prefix_sum_kernel(
    input_data: UnsafePointer[UInt32, MutAnyOrigin],
    output_prefix: UnsafePointer[UInt32, MutAnyOrigin],
    block_totals: UnsafePointer[UInt32, MutAnyOrigin],
    num_elements: UInt,
):
    """Compute exclusive prefix sum within each block, output block totals."""
    var tid = Int(thread_idx.x)
    var gid = Int(block_dim.x) * Int(block_idx.x) + tid
    var block_id = Int(block_idx.x)

    # Load input value (0 if out of bounds)
    var val: UInt32 = 0
    if gid < Int(num_elements):
        val = input_data[gid]

    # Use block prefix sum (exclusive)
    var prefix = block.prefix_sum[exclusive=True, block_size=BLOCK_SIZE_OPT](
        val
    )

    # Write prefix sum
    if gid < Int(num_elements):
        output_prefix[gid] = prefix

    # Last thread in block writes block total
    barrier()
    if tid == BLOCK_SIZE_OPT - 1 or gid == Int(num_elements) - 1:
        block_totals[block_id] = prefix + val


# ===== Kernel 3: Add block offsets to prefix sums =====
fn add_block_offsets_kernel(
    prefix_sums: UnsafePointer[UInt32, MutAnyOrigin],
    block_offsets: UnsafePointer[UInt32, MutAnyOrigin],
    num_elements: UInt,
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


# Character type encoding for bracket matching
comptime CHAR_TYPE_OPEN_BRACE: UInt8 = 1  # {
comptime CHAR_TYPE_CLOSE_BRACE: UInt8 = 2  # }
comptime CHAR_TYPE_OPEN_BRACKET: UInt8 = 3  # [
comptime CHAR_TYPE_CLOSE_BRACKET: UInt8 = 4  # ]
comptime CHAR_TYPE_OTHER: UInt8 = 0  # : or ,


# ===== Kernel 4: Scatter positions AND char types using bitmap and prefix offsets =====
fn scatter_positions_kernel(
    bitmap: UnsafePointer[UInt32, MutAnyOrigin],
    input_data: UnsafePointer[UInt8, MutAnyOrigin],
    prefix_offsets: UnsafePointer[UInt32, MutAnyOrigin],
    output_positions: UnsafePointer[Int32, MutAnyOrigin],
    output_char_types: UnsafePointer[UInt8, MutAnyOrigin],
    num_words: UInt,
    max_byte_pos: UInt,
):
    """Extract and scatter positions + char types from bitmap using prefix offsets.
    """
    var gid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if gid >= Int(num_words):
        return

    var bits = bitmap[gid]
    if bits == 0:
        return

    var base_pos = gid * 32
    var write_idx = Int(prefix_offsets[gid])

    # Extract positions using CTZ
    while bits != 0:
        # Count trailing zeros to find next set bit
        var tz = _ctz32_gpu(bits)
        var pos = base_pos + Int(tz)

        if pos < Int(max_byte_pos):
            output_positions[write_idx] = Int32(pos)

            # Read char and encode type for fast bracket matching
            var c = input_data[pos]
            var char_type: UInt8 = CHAR_TYPE_OTHER
            if c == 0x7B:  # {
                char_type = CHAR_TYPE_OPEN_BRACE
            elif c == 0x7D:  # }
                char_type = CHAR_TYPE_CLOSE_BRACE
            elif c == 0x5B:  # [
                char_type = CHAR_TYPE_OPEN_BRACKET
            elif c == 0x5D:  # ]
                char_type = CHAR_TYPE_CLOSE_BRACKET
            output_char_types[write_idx] = char_type

            write_idx += 1

        # Clear lowest set bit
        bits = bits & (bits - 1)


fn _ctz32_gpu(value: UInt32) -> UInt32:
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
fn _compute_block_prefix_sums(
    ctx: DeviceContext,
    d_block_totals_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    d_block_prefix_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    num_blocks: Int,
) raises:
    """Recursively compute prefix sum of block totals.

    Handles arbitrary levels of hierarchy.
    """
    if num_blocks <= BLOCK_SIZE_OPT:
        # Base case: single block can handle all totals
        var d_dummy = ctx.enqueue_create_buffer[DType.uint32](1)
        d_dummy.enqueue_fill(0)

        ctx.enqueue_function_unchecked[prefix_sum_kernel](
            d_block_totals_ptr,
            d_block_prefix_ptr,
            d_dummy.unsafe_ptr(),
            UInt(num_blocks),
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

    ctx.enqueue_function_unchecked[prefix_sum_kernel](
        d_block_totals_ptr,
        d_block_prefix_ptr,
        d_block_totals_l1.unsafe_ptr(),
        UInt(num_blocks),
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
        d_block_totals_l1.unsafe_ptr(),
        d_block_prefix_l1.unsafe_ptr(),
        num_blocks_l1,
    )

    # Propagate offsets back down
    ctx.enqueue_function_unchecked[add_block_offsets_kernel](
        d_block_prefix_ptr,
        d_block_prefix_l1.unsafe_ptr(),
        UInt(num_blocks),
        grid_dim=num_blocks_l1,
        block_dim=BLOCK_SIZE_OPT,
    )


# ===== Main function: GPU stream compaction =====
fn extract_positions_gpu(
    ctx: DeviceContext,
    d_bitmap_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    d_input_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    num_words: Int,
    max_byte_pos: Int,
) raises -> Tuple[List[Int32], List[UInt8], Int]:
    """Extract positions and char types from bitmap using GPU stream compaction.

    Args:
        ctx: GPU device context
        d_bitmap_ptr: Device pointer to structural bitmap
        d_input_ptr: Device pointer to input JSON data
        num_words: Number of 32-bit words in bitmap
        max_byte_pos: Maximum valid byte position

    Returns:
        Tuple of (list of positions, list of char types, total count)
    """
    var num_blocks = ceildiv(num_words, BLOCK_SIZE_OPT)

    # Phase 1: Compute popcounts
    var d_popcounts = ctx.enqueue_create_buffer[DType.uint32](num_words)
    ctx.enqueue_function_unchecked[popcount_kernel](
        d_bitmap_ptr,
        d_popcounts.unsafe_ptr(),
        UInt(num_words),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE_OPT,
    )

    # Phase 2: Prefix sum of popcounts (hierarchical)
    var d_prefix = ctx.enqueue_create_buffer[DType.uint32](num_words)
    var d_block_totals = ctx.enqueue_create_buffer[DType.uint32](num_blocks)
    d_block_totals.enqueue_fill(0)

    ctx.enqueue_function_unchecked[prefix_sum_kernel](
        d_popcounts.unsafe_ptr(),
        d_prefix.unsafe_ptr(),
        d_block_totals.unsafe_ptr(),
        UInt(num_words),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE_OPT,
    )

    # If multiple blocks, need to propagate block offsets recursively
    if num_blocks > 1:
        # Recursively compute prefix sum of block totals
        var d_block_prefix = ctx.enqueue_create_buffer[DType.uint32](num_blocks)
        d_block_prefix.enqueue_fill(0)

        _compute_block_prefix_sums(
            ctx,
            d_block_totals.unsafe_ptr(),
            d_block_prefix.unsafe_ptr(),
            num_blocks,
        )

        # Add block offsets to element-level prefix sums
        ctx.enqueue_function_unchecked[add_block_offsets_kernel](
            d_prefix.unsafe_ptr(),
            d_block_prefix.unsafe_ptr(),
            UInt(num_words),
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE_OPT,
        )

    # Get total count - copy last prefix and popcount to compute total
    ctx.synchronize()

    var h_last = ctx.enqueue_create_host_buffer[DType.uint32](2)

    # Copy last elements using a simple kernel
    fn copy_last_kernel(
        prefix: UnsafePointer[UInt32, MutAnyOrigin],
        popcount: UnsafePointer[UInt32, MutAnyOrigin],
        output: UnsafePointer[UInt32, MutAnyOrigin],
        last_idx: UInt,
    ):
        if Int(thread_idx.x) == 0:
            output[0] = prefix[Int(last_idx)]
            output[1] = popcount[Int(last_idx)]

    var d_last = ctx.enqueue_create_buffer[DType.uint32](2)
    ctx.enqueue_function_unchecked[copy_last_kernel](
        d_prefix.unsafe_ptr(),
        d_popcounts.unsafe_ptr(),
        d_last.unsafe_ptr(),
        UInt(num_words - 1),
        grid_dim=1,
        block_dim=1,
    )

    ctx.enqueue_copy(h_last, d_last)
    ctx.synchronize()

    var total_count = Int(h_last.unsafe_ptr()[0] + h_last.unsafe_ptr()[1])

    if total_count == 0:
        return (List[Int32](), List[UInt8](), 0)

    # Phase 3: Scatter positions and char types
    var d_positions = ctx.enqueue_create_buffer[DType.int32](total_count)
    var d_char_types = ctx.enqueue_create_buffer[DType.uint8](total_count)
    d_positions.enqueue_fill(0)
    d_char_types.enqueue_fill(0)

    ctx.enqueue_function_unchecked[scatter_positions_kernel](
        d_bitmap_ptr,
        d_input_ptr,
        d_prefix.unsafe_ptr(),
        d_positions.unsafe_ptr(),
        d_char_types.unsafe_ptr(),
        UInt(num_words),
        UInt(max_byte_pos),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE_OPT,
    )

    # Copy back to host
    var h_positions = ctx.enqueue_create_host_buffer[DType.int32](total_count)
    var h_char_types = ctx.enqueue_create_host_buffer[DType.uint8](total_count)
    ctx.enqueue_copy(h_positions, d_positions)
    ctx.enqueue_copy(h_char_types, d_char_types)
    ctx.synchronize()

    # Convert to Lists
    var positions = List[Int32](capacity=total_count)
    positions.resize(total_count, 0)
    memcpy(
        dest=positions.unsafe_ptr(),
        src=h_positions.unsafe_ptr(),
        count=total_count,
    )

    var char_types = List[UInt8](capacity=total_count)
    char_types.resize(total_count, 0)
    memcpy(
        dest=char_types.unsafe_ptr(),
        src=h_char_types.unsafe_ptr(),
        count=total_count,
    )

    return (positions^, char_types^, total_count)
