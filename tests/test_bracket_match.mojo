# Tests for GPU bracket matching

from testing import assert_equal, assert_true
from gpu.host import DeviceContext
from collections import List
from memory import memcpy

from src.gpu.bracket_match import match_brackets_gpu


fn make_char_list(chars: String) -> List[UInt8]:
    """Create a list of char types from a string."""
    var result = List[UInt8](capacity=len(chars))
    for i in range(len(chars)):
        result.append(ord(chars[i]))
    return result^


fn test_simple_braces() raises:
    """Test simple case: {}."""
    var ctx = DeviceContext()

    # Input: { }
    var char_types = make_char_list("{}")
    var n = len(char_types)

    var d_char_types = ctx.enqueue_create_buffer[DType.uint8](n)
    var h_char_types = ctx.enqueue_create_host_buffer[DType.uint8](n)
    memcpy(dest=h_char_types.unsafe_ptr(), src=char_types.unsafe_ptr(), count=n)
    ctx.enqueue_copy(d_char_types, h_char_types)
    ctx.synchronize()

    var result = match_brackets_gpu(ctx, d_char_types.unsafe_ptr(), n)
    var depths = result[0].copy()
    var pair_pos = result[1].copy().copy()

    # After adjustment: { has depth 0, } has depth 0
    assert_equal(Int(depths[0]), 0)  # { adjusted from 1 to 0
    assert_equal(Int(depths[1]), 0)  # }

    # Pairing: { at 0 matches } at 1
    assert_equal(Int(pair_pos[0]), 1)
    assert_equal(Int(pair_pos[1]), 0)

    print("PASS: test_simple_braces")


fn test_nested_braces() raises:
    """Test nested: {{}}."""
    var ctx = DeviceContext()

    var char_types = make_char_list("{{}}")
    var n = len(char_types)

    var d_char_types = ctx.enqueue_create_buffer[DType.uint8](n)
    var h_char_types = ctx.enqueue_create_host_buffer[DType.uint8](n)
    memcpy(dest=h_char_types.unsafe_ptr(), src=char_types.unsafe_ptr(), count=n)
    ctx.enqueue_copy(d_char_types, h_char_types)
    ctx.synchronize()

    var result = match_brackets_gpu(ctx, d_char_types.unsafe_ptr(), n)
    var depths = result[0].copy()
    var pair_pos = result[1].copy().copy()

    # Depths after adjustment:
    # { (pos 0): depth 0 (was 1, adjusted)
    # { (pos 1): depth 1 (was 2, adjusted)
    # } (pos 2): depth 1
    # } (pos 3): depth 0
    assert_equal(Int(depths[0]), 0)
    assert_equal(Int(depths[1]), 1)
    assert_equal(Int(depths[2]), 1)
    assert_equal(Int(depths[3]), 0)

    # Pairing:
    # { at 0 matches } at 3 (both depth 0)
    # { at 1 matches } at 2 (both depth 1)
    assert_equal(Int(pair_pos[0]), 3)
    assert_equal(Int(pair_pos[1]), 2)
    assert_equal(Int(pair_pos[2]), 1)
    assert_equal(Int(pair_pos[3]), 0)

    print("PASS: test_nested_braces")


fn test_mixed_brackets() raises:
    """Test mixed: {[]}."""
    var ctx = DeviceContext()

    var char_types = make_char_list("{[]}")
    var n = len(char_types)

    var d_char_types = ctx.enqueue_create_buffer[DType.uint8](n)
    var h_char_types = ctx.enqueue_create_host_buffer[DType.uint8](n)
    memcpy(dest=h_char_types.unsafe_ptr(), src=char_types.unsafe_ptr(), count=n)
    ctx.enqueue_copy(d_char_types, h_char_types)
    ctx.synchronize()

    var result = match_brackets_gpu(ctx, d_char_types.unsafe_ptr(), n)
    var pair_pos = result[1].copy()

    # { at 0 matches } at 3
    # [ at 1 matches ] at 2
    assert_equal(Int(pair_pos[0]), 3)
    assert_equal(Int(pair_pos[1]), 2)
    assert_equal(Int(pair_pos[2]), 1)
    assert_equal(Int(pair_pos[3]), 0)

    print("PASS: test_mixed_brackets")


fn test_with_other_chars() raises:
    """Test with colons and commas: {:,}."""
    var ctx = DeviceContext()

    var char_types = make_char_list("{:,}")
    var n = len(char_types)

    var d_char_types = ctx.enqueue_create_buffer[DType.uint8](n)
    var h_char_types = ctx.enqueue_create_host_buffer[DType.uint8](n)
    memcpy(dest=h_char_types.unsafe_ptr(), src=char_types.unsafe_ptr(), count=n)
    ctx.enqueue_copy(d_char_types, h_char_types)
    ctx.synchronize()

    var result = match_brackets_gpu(ctx, d_char_types.unsafe_ptr(), n)
    var pair_pos = result[1].copy()

    # { at 0 matches } at 3
    # : and , are not brackets, pair_pos = -1
    assert_equal(Int(pair_pos[0]), 3)
    assert_equal(Int(pair_pos[1]), -1)
    assert_equal(Int(pair_pos[2]), -1)
    assert_equal(Int(pair_pos[3]), 0)

    print("PASS: test_with_other_chars")


fn test_deeply_nested() raises:
    """Test deeply nested: {{{}}}."""
    var ctx = DeviceContext()

    var char_types = make_char_list("{{{}}}")
    var n = len(char_types)

    var d_char_types = ctx.enqueue_create_buffer[DType.uint8](n)
    var h_char_types = ctx.enqueue_create_host_buffer[DType.uint8](n)
    memcpy(dest=h_char_types.unsafe_ptr(), src=char_types.unsafe_ptr(), count=n)
    ctx.enqueue_copy(d_char_types, h_char_types)
    ctx.synchronize()

    var result = match_brackets_gpu(ctx, d_char_types.unsafe_ptr(), n)
    var depths = result[0].copy()
    var pair_pos = result[1].copy().copy()

    # Depths after adjustment: 0, 1, 2, 2, 1, 0
    assert_equal(Int(depths[0]), 0)
    assert_equal(Int(depths[1]), 1)
    assert_equal(Int(depths[2]), 2)
    assert_equal(Int(depths[3]), 2)
    assert_equal(Int(depths[4]), 1)
    assert_equal(Int(depths[5]), 0)

    # Pairing:
    # { at 0 matches } at 5
    # { at 1 matches } at 4
    # { at 2 matches } at 3
    assert_equal(Int(pair_pos[0]), 5)
    assert_equal(Int(pair_pos[1]), 4)
    assert_equal(Int(pair_pos[2]), 3)
    assert_equal(Int(pair_pos[3]), 2)
    assert_equal(Int(pair_pos[4]), 1)
    assert_equal(Int(pair_pos[5]), 0)

    print("PASS: test_deeply_nested")


fn test_sibling_objects() raises:
    """Test sibling objects: {}{}."""
    var ctx = DeviceContext()

    var char_types = make_char_list("{}{}")
    var n = len(char_types)

    var d_char_types = ctx.enqueue_create_buffer[DType.uint8](n)
    var h_char_types = ctx.enqueue_create_host_buffer[DType.uint8](n)
    memcpy(dest=h_char_types.unsafe_ptr(), src=char_types.unsafe_ptr(), count=n)
    ctx.enqueue_copy(d_char_types, h_char_types)
    ctx.synchronize()

    var result = match_brackets_gpu(ctx, d_char_types.unsafe_ptr(), n)
    var pair_pos = result[1].copy()

    # { at 0 matches } at 1
    # { at 2 matches } at 3
    assert_equal(Int(pair_pos[0]), 1)
    assert_equal(Int(pair_pos[1]), 0)
    assert_equal(Int(pair_pos[2]), 3)
    assert_equal(Int(pair_pos[3]), 2)

    print("PASS: test_sibling_objects")


fn main() raises:
    print("=== GPU Bracket Matching Tests ===")
    print()

    test_simple_braces()
    test_nested_braces()
    test_mixed_brackets()
    test_with_other_chars()
    test_deeply_nested()
    test_sibling_objects()

    print()
    print("All bracket matching tests passed!")
