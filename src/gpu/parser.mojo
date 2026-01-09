# Optimized JSON Parser - GPU-accelerated with fully parallel prefix sums
#
# Key optimizations:
# 1. Fully GPU-parallel hierarchical prefix sums (no CPU loop!)
# 2. Pinned memory for faster H2D transfers
# 3. Minimal host-device synchronization
# 4. Fused kernels

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu import block_dim, block_idx, thread_idx, barrier, block, global_idx
from gpu.memory import AddressSpace
from collections import List
from memory import UnsafePointer, memcpy
from math import ceildiv

from ..types import JSONInput, JSONResult
from .kernels import (
    BLOCK_SIZE_OPT,
    popcount_fast,
    fused_json_kernel,
    structural_popcount_kernel,
    extract_positions_kernel,
)
from .stream_compact import extract_positions_gpu


fn parse_json_gpu(var input: JSONInput) raises -> JSONResult:
    """GPU JSON parsing with parallel algorithms."""
    var size = len(input.data)

    if size == 0:
        var result = JSONResult()
        return result^

    var total_padded_32 = (size + 31) // 32

    # Always use GPU (assume it's available)
    var ctx = DeviceContext()
    return _parse_json_gpu_optimized(ctx, input^, size, total_padded_32)


fn parse_json_gpu_from_pinned(
    ctx: DeviceContext,
    h_input: HostBuffer[DType.uint8],
    size: Int,
) raises -> JSONResult:
    """GPU JSON parsing from pre-loaded pinned memory (fastest path).

    This skips the memcpy to pinned memory, saving ~100ms for large files.
    Use this when you can read the file directly into pinned memory.
    """
    if size == 0:
        var result = JSONResult()
        return result^

    var total_padded_32 = (size + 31) // 32
    return _parse_json_gpu_from_pinned_impl(ctx, h_input, size, total_padded_32)


comptime DEBUG_TIMING: Bool = False


fn _parse_json_gpu_optimized(
    ctx: DeviceContext, var input: JSONInput, size: Int, total_padded_32: Int
) raises -> JSONResult:
    """GPU-accelerated JSON parsing with fully parallel prefix sums."""
    from time import perf_counter_ns

    var result = JSONResult()
    result.file_size = size

    # Calculate grid dimensions
    var num_blocks = ceildiv(total_padded_32, BLOCK_SIZE_OPT)
    if num_blocks == 0:
        num_blocks = 1

    var t0 = perf_counter_ns()

    # ===== Phase 1: Allocate and transfer input =====
    var d_input = ctx.enqueue_create_buffer[DType.uint8](size)

    # Use pinned host buffer for faster transfer
    var h_input = ctx.enqueue_create_host_buffer[DType.uint8](size)

    @parameter
    if DEBUG_TIMING:
        ctx.synchronize()
        var t_alloc = perf_counter_ns()
        print("    Buffer alloc:", Float64(t_alloc - t0) / 1e6, "ms")

    memcpy(dest=h_input.unsafe_ptr(), src=input.data.unsafe_ptr(), count=size)

    @parameter
    if DEBUG_TIMING:
        var t_memcpy = perf_counter_ns()
        print("    memcpy:", Float64(t_memcpy - t0) / 1e6, "ms (cumulative)")

    ctx.enqueue_copy(d_input, h_input)

    @parameter
    if DEBUG_TIMING:
        ctx.synchronize()
        var t_h2d = perf_counter_ns()
        print("    H2D transfer:", Float64(t_h2d - t0) / 1e6, "ms (cumulative)")

    # Allocate output buffers
    var d_quote_bitmap = ctx.enqueue_create_buffer[DType.uint32](
        total_padded_32
    )
    var d_popcounts = ctx.enqueue_create_buffer[DType.uint32](total_padded_32)
    var d_quote_prefix = ctx.enqueue_create_buffer[DType.uint32](
        total_padded_32
    )
    var d_structural = ctx.enqueue_create_buffer[DType.uint32](total_padded_32)
    var d_open_close = ctx.enqueue_create_buffer[DType.uint32](total_padded_32)

    # Note: Skipping enqueue_fill(0) - kernels overwrite all values

    # ===== Phase 2: Build quote bitmap and compute popcounts =====
    ctx.enqueue_function_unchecked[_quote_popcount_kernel](
        d_input.unsafe_ptr(),
        d_quote_bitmap.unsafe_ptr(),
        d_popcounts.unsafe_ptr(),
        UInt(size),
        UInt(total_padded_32),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE_OPT,
    )

    # ===== Phase 3: Hierarchical GPU-parallel prefix sums =====
    # Level 0: Block-local prefix sums + block totals
    var d_block_sums = ctx.enqueue_create_buffer[DType.uint32](num_blocks)

    ctx.enqueue_function_unchecked[_block_prefix_kernel](
        d_popcounts.unsafe_ptr(),
        d_quote_prefix.unsafe_ptr(),
        d_block_sums.unsafe_ptr(),
        UInt(total_padded_32),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE_OPT,
    )

    # Level 1+: If many blocks, compute prefix sums of block sums recursively
    if num_blocks > 1:
        # Allocate buffers for block prefix sums
        var d_block_prefix = ctx.enqueue_create_buffer[DType.uint32](num_blocks)

        var num_blocks_l1 = ceildiv(num_blocks, BLOCK_SIZE_OPT)

        if num_blocks_l1 == 1:
            # Single block can handle all block sums
            var d_dummy = ctx.enqueue_create_buffer[DType.uint32](1)

            ctx.enqueue_function_unchecked[_block_prefix_kernel](
                d_block_sums.unsafe_ptr(),
                d_block_prefix.unsafe_ptr(),
                d_dummy.unsafe_ptr(),
                UInt(num_blocks),
                grid_dim=1,
                block_dim=min(num_blocks, BLOCK_SIZE_OPT),
            )
        else:
            # Need another level of hierarchy
            var d_block_sums_l1 = ctx.enqueue_create_buffer[DType.uint32](
                num_blocks_l1
            )

            ctx.enqueue_function_unchecked[_block_prefix_kernel](
                d_block_sums.unsafe_ptr(),
                d_block_prefix.unsafe_ptr(),
                d_block_sums_l1.unsafe_ptr(),
                UInt(num_blocks),
                grid_dim=num_blocks_l1,
                block_dim=BLOCK_SIZE_OPT,
            )

            # Level 2: prefix sum of level 1 block sums
            var num_blocks_l2 = ceildiv(num_blocks_l1, BLOCK_SIZE_OPT)
            var d_block_prefix_l1 = ctx.enqueue_create_buffer[DType.uint32](
                num_blocks_l1
            )

            var d_dummy = ctx.enqueue_create_buffer[DType.uint32](1)

            ctx.enqueue_function_unchecked[_block_prefix_kernel](
                d_block_sums_l1.unsafe_ptr(),
                d_block_prefix_l1.unsafe_ptr(),
                d_dummy.unsafe_ptr(),
                UInt(num_blocks_l1),
                grid_dim=num_blocks_l2,
                block_dim=min(num_blocks_l1, BLOCK_SIZE_OPT),
            )

            # Add L1 prefix back to L0 prefix
            ctx.enqueue_function_unchecked[_add_block_offset_kernel](
                d_block_prefix.unsafe_ptr(),
                d_block_prefix_l1.unsafe_ptr(),
                UInt(num_blocks),
                grid_dim=num_blocks_l1,
                block_dim=BLOCK_SIZE_OPT,
            )

        # Add block prefixes to element-level prefix sums
        ctx.enqueue_function_unchecked[_add_block_offset_kernel](
            d_quote_prefix.unsafe_ptr(),
            d_block_prefix.unsafe_ptr(),
            UInt(total_padded_32),
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE_OPT,
        )

    # ===== Phase 4: Main fused kernel (uses prefix sums for in-string detection) =====
    ctx.enqueue_function_unchecked[fused_json_kernel](
        d_input.unsafe_ptr(),
        d_structural.unsafe_ptr(),
        d_open_close.unsafe_ptr(),
        d_quote_prefix.unsafe_ptr(),
        UInt(size),
        UInt(total_padded_32),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE_OPT,
    )

    # Single sync point before copying back
    ctx.synchronize()
    var t1 = perf_counter_ns()

    @parameter
    if DEBUG_TIMING:
        print("  H2D + GPU kernels:", Float64(t1 - t0) / 1e6, "ms")

    # ===== Phase 5: Extract positions from structural bitmap (GPU stream compaction) =====
    var gpu_result = extract_positions_gpu(
        ctx,
        d_structural.unsafe_ptr(),
        d_input.unsafe_ptr(),
        total_padded_32,
        size,
    )
    result.structural = gpu_result[0].copy()
    var char_types = gpu_result[1].copy()
    var count = gpu_result[2]
    result.pair_pos = List[Int32](capacity=count)
    result.pair_pos.resize(count, -1)

    @parameter
    if DEBUG_TIMING:
        var t2 = perf_counter_ns()
        print("  Position extraction:", Float64(t2 - t1) / 1e6, "ms")
        print("  Structural count:", len(result.structural))

    # Match brackets on CPU using pre-computed char types
    _match_brackets_fast(result, char_types)

    @parameter
    if DEBUG_TIMING:
        var t4 = perf_counter_ns()
        print("  Bracket matching:", Float64(t4 - t1) / 1e6, "ms")
        print("  TOTAL GPU parse:", Float64(t4 - t0) / 1e6, "ms")

    return result^


fn _parse_json_gpu_from_pinned_impl(
    ctx: DeviceContext,
    h_input: HostBuffer[DType.uint8],
    size: Int,
    total_padded_32: Int,
) raises -> JSONResult:
    """GPU parsing from pre-loaded pinned memory - skips memcpy overhead."""
    from time import perf_counter_ns

    var result = JSONResult()
    result.file_size = size

    var num_blocks = ceildiv(total_padded_32, BLOCK_SIZE_OPT)
    if num_blocks == 0:
        num_blocks = 1

    var t0 = perf_counter_ns()

    # Direct H2D from pinned memory - no memcpy!
    var d_input = ctx.enqueue_create_buffer[DType.uint8](size)
    ctx.enqueue_copy(d_input, h_input)

    @parameter
    if DEBUG_TIMING:
        ctx.synchronize()
        var t_h2d = perf_counter_ns()
        print("    H2D (from pinned):", Float64(t_h2d - t0) / 1e6, "ms")

    # Allocate output buffers
    var d_quote_bitmap = ctx.enqueue_create_buffer[DType.uint32](
        total_padded_32
    )
    var d_popcounts = ctx.enqueue_create_buffer[DType.uint32](total_padded_32)
    var d_quote_prefix = ctx.enqueue_create_buffer[DType.uint32](
        total_padded_32
    )
    var d_structural = ctx.enqueue_create_buffer[DType.uint32](total_padded_32)
    var d_open_close = ctx.enqueue_create_buffer[DType.uint32](total_padded_32)

    # Phase 2: Build quote bitmap and compute popcounts
    ctx.enqueue_function_unchecked[_quote_popcount_kernel](
        d_input.unsafe_ptr(),
        d_quote_bitmap.unsafe_ptr(),
        d_popcounts.unsafe_ptr(),
        UInt(size),
        UInt(total_padded_32),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE_OPT,
    )

    # Phase 3: Hierarchical prefix sums
    var d_block_sums = ctx.enqueue_create_buffer[DType.uint32](num_blocks)

    ctx.enqueue_function_unchecked[_block_prefix_kernel](
        d_popcounts.unsafe_ptr(),
        d_quote_prefix.unsafe_ptr(),
        d_block_sums.unsafe_ptr(),
        UInt(total_padded_32),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE_OPT,
    )

    if num_blocks > 1:
        var d_block_prefix = ctx.enqueue_create_buffer[DType.uint32](num_blocks)
        var num_blocks_l1 = ceildiv(num_blocks, BLOCK_SIZE_OPT)

        if num_blocks_l1 == 1:
            var d_dummy = ctx.enqueue_create_buffer[DType.uint32](1)
            ctx.enqueue_function_unchecked[_block_prefix_kernel](
                d_block_sums.unsafe_ptr(),
                d_block_prefix.unsafe_ptr(),
                d_dummy.unsafe_ptr(),
                UInt(num_blocks),
                grid_dim=1,
                block_dim=min(num_blocks, BLOCK_SIZE_OPT),
            )
        else:
            var d_block_sums_l1 = ctx.enqueue_create_buffer[DType.uint32](
                num_blocks_l1
            )
            ctx.enqueue_function_unchecked[_block_prefix_kernel](
                d_block_sums.unsafe_ptr(),
                d_block_prefix.unsafe_ptr(),
                d_block_sums_l1.unsafe_ptr(),
                UInt(num_blocks),
                grid_dim=num_blocks_l1,
                block_dim=BLOCK_SIZE_OPT,
            )

            var num_blocks_l2 = ceildiv(num_blocks_l1, BLOCK_SIZE_OPT)
            var d_block_prefix_l1 = ctx.enqueue_create_buffer[DType.uint32](
                num_blocks_l1
            )
            var d_dummy = ctx.enqueue_create_buffer[DType.uint32](1)

            ctx.enqueue_function_unchecked[_block_prefix_kernel](
                d_block_sums_l1.unsafe_ptr(),
                d_block_prefix_l1.unsafe_ptr(),
                d_dummy.unsafe_ptr(),
                UInt(num_blocks_l1),
                grid_dim=num_blocks_l2,
                block_dim=min(num_blocks_l1, BLOCK_SIZE_OPT),
            )

            ctx.enqueue_function_unchecked[_add_block_offset_kernel](
                d_block_prefix.unsafe_ptr(),
                d_block_prefix_l1.unsafe_ptr(),
                UInt(num_blocks),
                grid_dim=num_blocks_l1,
                block_dim=BLOCK_SIZE_OPT,
            )

        ctx.enqueue_function_unchecked[_add_block_offset_kernel](
            d_quote_prefix.unsafe_ptr(),
            d_block_prefix.unsafe_ptr(),
            UInt(total_padded_32),
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE_OPT,
        )

    # Phase 4: Main fused kernel
    ctx.enqueue_function_unchecked[fused_json_kernel](
        d_input.unsafe_ptr(),
        d_structural.unsafe_ptr(),
        d_open_close.unsafe_ptr(),
        d_quote_prefix.unsafe_ptr(),
        UInt(size),
        UInt(total_padded_32),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE_OPT,
    )

    ctx.synchronize()
    var t1 = perf_counter_ns()

    @parameter
    if DEBUG_TIMING:
        print("  H2D + GPU kernels:", Float64(t1 - t0) / 1e6, "ms")

    # Phase 5: Extract positions using GPU stream compaction
    var char_types = List[UInt8]()
    var gpu_result = extract_positions_gpu(
        ctx,
        d_structural.unsafe_ptr(),
        d_input.unsafe_ptr(),
        total_padded_32,
        size,
    )
    result.structural = gpu_result[0].copy()
    char_types = gpu_result[1].copy()
    var count = gpu_result[2]
    result.pair_pos = List[Int32](capacity=count)
    result.pair_pos.resize(count, -1)

    @parameter
    if DEBUG_TIMING:
        var t2 = perf_counter_ns()
        print("  Position extraction:", Float64(t2 - t1) / 1e6, "ms")

    # Match brackets on CPU
    _match_brackets_fast(result, char_types)

    @parameter
    if DEBUG_TIMING:
        var t4 = perf_counter_ns()
        print("  Bracket matching:", Float64(t4 - t1) / 1e6, "ms")
        print("  TOTAL GPU parse (from pinned):", Float64(t4 - t0) / 1e6, "ms")

    return result^


# ===== GPU Kernels =====


fn _quote_popcount_kernel(
    input_data: UnsafePointer[UInt8, MutAnyOrigin],
    output_quote: UnsafePointer[UInt32, MutAnyOrigin],
    output_popcount: UnsafePointer[UInt32, MutAnyOrigin],
    size: UInt,
    total_padded_32: UInt,
):
    """Extract quote bitmap and compute popcount in one pass."""
    var gid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)

    if gid >= Int(total_padded_32):
        return

    var start_pos = gid * 32
    var quote_bits: UInt32 = 0
    var slash_bits: UInt32 = 0

    for j in range(32):
        var pos = start_pos + j
        if pos >= Int(size):
            break

        var c = input_data[pos]
        var bit_mask = UInt32(1) << UInt32(j)

        if c == 0x22:  # quote
            quote_bits |= bit_mask
        if c == 0x5C:  # backslash
            slash_bits |= bit_mask

    # Simple escape detection
    var escaped = quote_bits & (slash_bits << 1)
    var real_quotes = quote_bits & (~escaped)

    output_quote[gid] = real_quotes
    output_popcount[gid] = popcount_fast(real_quotes)


fn _block_prefix_kernel(
    input_data: UnsafePointer[UInt32, MutAnyOrigin],
    output_prefix: UnsafePointer[UInt32, MutAnyOrigin],
    block_sums: UnsafePointer[UInt32, MutAnyOrigin],
    total_size: UInt,
):
    """Compute block-local exclusive prefix sum and output block total."""
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var gid = bid * Int(block_dim.x) + tid

    # Load value (0 if out of bounds)
    var val: UInt32 = 0
    if gid < Int(total_size):
        val = input_data[gid]

    # Compute block-local exclusive prefix sum using GPU primitive
    var prefix = block.prefix_sum[exclusive=True, block_size=BLOCK_SIZE_OPT](
        val
    )

    # Write prefix sum back
    if gid < Int(total_size):
        output_prefix[gid] = prefix

    # Last thread in block writes block total
    var block_end = min((bid + 1) * Int(block_dim.x), Int(total_size))
    var last_in_block = block_end - 1 - bid * Int(block_dim.x)

    if tid == last_in_block:
        block_sums[bid] = prefix + val


fn _add_block_offset_kernel(
    data: UnsafePointer[UInt32, MutAnyOrigin],
    block_offsets: UnsafePointer[UInt32, MutAnyOrigin],
    total_size: UInt,
):
    """Add block offset to each element."""
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var gid = bid * Int(block_dim.x) + tid

    if gid >= Int(total_size):
        return

    if bid > 0:
        data[gid] = data[gid] + block_offsets[bid]


# Char type constants (must match _gpu_stream_compact.mojo)
comptime CHAR_TYPE_OPEN_BRACE: UInt8 = 1  # {
comptime CHAR_TYPE_CLOSE_BRACE: UInt8 = 2  # }
comptime CHAR_TYPE_OPEN_BRACKET: UInt8 = 3  # [
comptime CHAR_TYPE_CLOSE_BRACKET: UInt8 = 4  # ]


fn _match_brackets_fast(mut result: JSONResult, char_types: List[UInt8]):
    """Match brackets using pre-computed char types (no memory reads)."""
    var stack = List[Int]()
    var n = len(result.structural)

    for i in range(n):
        var ct = char_types[i]

        # Open brackets: { or [
        if ct == CHAR_TYPE_OPEN_BRACE or ct == CHAR_TYPE_OPEN_BRACKET:
            stack.append(i)
        # Close brackets: } or ]
        elif ct == CHAR_TYPE_CLOSE_BRACE or ct == CHAR_TYPE_CLOSE_BRACKET:
            if len(stack) > 0:
                var open_idx = stack.pop()
                result.pair_pos[open_idx] = Int32(i)
                result.pair_pos[i] = Int32(open_idx)


