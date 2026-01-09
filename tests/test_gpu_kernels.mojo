# Tests for GPU kernels - stream compaction and bracket matching

from testing import assert_equal, assert_true
from gpu.host import DeviceContext
from collections import List
from memory import memcpy


fn test_stream_compact_simple() raises:
    """Test stream compaction with simple bitmap."""
    from src.gpu.stream_compact import extract_positions_gpu

    var ctx = DeviceContext()

    # Create a simple bitmap: positions 0, 5 are set
    # Word 0: bit 0, bit 5 -> 0b00100001 = 33
    var num_words = 1
    var h_bitmap = ctx.enqueue_create_host_buffer[DType.uint32](num_words)
    h_bitmap.unsafe_ptr()[0] = 33  # bits 0 and 5 set

    # Create dummy input data (just need bytes at positions 0, 5)
    var h_input = ctx.enqueue_create_host_buffer[DType.uint8](32)
    h_input.unsafe_ptr()[0] = 0x7B  # {
    h_input.unsafe_ptr()[5] = 0x7D  # }

    var d_bitmap = ctx.enqueue_create_buffer[DType.uint32](num_words)
    var d_input = ctx.enqueue_create_buffer[DType.uint8](32)
    ctx.enqueue_copy(d_bitmap, h_bitmap)
    ctx.enqueue_copy(d_input, h_input)
    ctx.synchronize()

    var result = extract_positions_gpu(
        ctx, d_bitmap.unsafe_ptr(), d_input.unsafe_ptr(), num_words, 32
    )
    var positions = result[0].copy()
    var count = result[2]

    assert_equal(count, 2)
    assert_equal(Int(positions[0]), 0)
    assert_equal(Int(positions[1]), 5)

    print("PASS: test_stream_compact_simple")


fn test_stream_compact_multiple_words() raises:
    """Test stream compaction with multiple bitmap words."""
    from src.gpu.stream_compact import extract_positions_gpu

    var ctx = DeviceContext()

    # Create bitmap with positions in multiple words
    var num_words = 3
    var h_bitmap = ctx.enqueue_create_host_buffer[DType.uint32](num_words)
    h_bitmap.unsafe_ptr()[0] = 1  # bit 0 -> position 0
    h_bitmap.unsafe_ptr()[1] = 1  # bit 0 -> position 32
    h_bitmap.unsafe_ptr()[2] = 32  # bit 5 -> position 69

    # Create dummy input data
    var h_input = ctx.enqueue_create_host_buffer[DType.uint8](100)
    h_input.unsafe_ptr()[0] = 0x7B  # {
    h_input.unsafe_ptr()[32] = 0x5B  # [
    h_input.unsafe_ptr()[69] = 0x5D  # ]

    var d_bitmap = ctx.enqueue_create_buffer[DType.uint32](num_words)
    var d_input = ctx.enqueue_create_buffer[DType.uint8](100)
    ctx.enqueue_copy(d_bitmap, h_bitmap)
    ctx.enqueue_copy(d_input, h_input)
    ctx.synchronize()

    var result = extract_positions_gpu(
        ctx, d_bitmap.unsafe_ptr(), d_input.unsafe_ptr(), num_words, 100
    )
    var positions = result[0].copy()
    var count = result[2]

    assert_equal(count, 3)
    assert_equal(Int(positions[0]), 0)
    assert_equal(Int(positions[1]), 32)
    assert_equal(Int(positions[2]), 69)

    print("PASS: test_stream_compact_multiple_words")


fn test_stream_compact_empty() raises:
    """Test stream compaction with empty bitmap."""
    from src.gpu.stream_compact import extract_positions_gpu

    var ctx = DeviceContext()

    var num_words = 4
    var h_bitmap = ctx.enqueue_create_host_buffer[DType.uint32](num_words)
    h_bitmap.unsafe_ptr()[0] = 0
    h_bitmap.unsafe_ptr()[1] = 0
    h_bitmap.unsafe_ptr()[2] = 0
    h_bitmap.unsafe_ptr()[3] = 0

    var h_input = ctx.enqueue_create_host_buffer[DType.uint8](128)

    var d_bitmap = ctx.enqueue_create_buffer[DType.uint32](num_words)
    var d_input = ctx.enqueue_create_buffer[DType.uint8](128)
    ctx.enqueue_copy(d_bitmap, h_bitmap)
    ctx.enqueue_copy(d_input, h_input)
    ctx.synchronize()

    var result = extract_positions_gpu(
        ctx, d_bitmap.unsafe_ptr(), d_input.unsafe_ptr(), num_words, 128
    )
    var count = result[2]

    assert_equal(count, 0)

    print("PASS: test_stream_compact_empty")


fn test_stream_compact_all_set() raises:
    """Test stream compaction with all bits set in one word."""
    from src.gpu.stream_compact import extract_positions_gpu

    var ctx = DeviceContext()

    var num_words = 1
    var h_bitmap = ctx.enqueue_create_host_buffer[DType.uint32](num_words)
    h_bitmap.unsafe_ptr()[0] = 0xFFFFFFFF  # All 32 bits set

    var h_input = ctx.enqueue_create_host_buffer[DType.uint8](32)
    for i in range(32):
        h_input.unsafe_ptr()[i] = 0x2C  # comma

    var d_bitmap = ctx.enqueue_create_buffer[DType.uint32](num_words)
    var d_input = ctx.enqueue_create_buffer[DType.uint8](32)
    ctx.enqueue_copy(d_bitmap, h_bitmap)
    ctx.enqueue_copy(d_input, h_input)
    ctx.synchronize()

    var result = extract_positions_gpu(
        ctx, d_bitmap.unsafe_ptr(), d_input.unsafe_ptr(), num_words, 32
    )
    var positions = result[0].copy()
    var count = result[2]

    assert_equal(count, 32)

    # Check all positions are correct
    for i in range(32):
        assert_equal(Int(positions[i]), i)

    print("PASS: test_stream_compact_all_set")


fn test_stream_compact_large() raises:
    """Test stream compaction with larger bitmap (multiple blocks)."""
    from src.gpu.stream_compact import extract_positions_gpu

    var ctx = DeviceContext()

    # Create 1024 words = 32KB of bitmap
    var num_words = 1024
    var max_pos = num_words * 32
    var h_bitmap = ctx.enqueue_create_host_buffer[DType.uint32](num_words)

    # Set bit 0 in every word -> 1024 positions
    for i in range(num_words):
        h_bitmap.unsafe_ptr()[i] = 1

    # Create dummy input data
    var h_input = ctx.enqueue_create_host_buffer[DType.uint8](max_pos)
    for i in range(num_words):
        h_input.unsafe_ptr()[i * 32] = 0x3A  # colon

    var d_bitmap = ctx.enqueue_create_buffer[DType.uint32](num_words)
    var d_input = ctx.enqueue_create_buffer[DType.uint8](max_pos)
    ctx.enqueue_copy(d_bitmap, h_bitmap)
    ctx.enqueue_copy(d_input, h_input)
    ctx.synchronize()

    var result = extract_positions_gpu(
        ctx, d_bitmap.unsafe_ptr(), d_input.unsafe_ptr(), num_words, max_pos
    )
    var positions = result[0].copy()
    var count = result[2]

    assert_equal(count, 1024)

    # Check positions are correct (every 32nd position)
    for i in range(1024):
        assert_equal(Int(positions[i]), i * 32)

    print("PASS: test_stream_compact_large")


fn main() raises:
    print("=== GPU Kernel Tests ===")
    print()

    print("--- Stream Compaction Tests ---")
    test_stream_compact_simple()
    test_stream_compact_multiple_words()
    test_stream_compact_empty()
    test_stream_compact_all_set()
    test_stream_compact_large()

    print()
    print("All GPU kernel tests passed!")
