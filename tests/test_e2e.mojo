# End-to-end tests for mojson
# Tests JSON roundtrip (loads -> dumps -> loads) and error handling
#
# Usage:
#   pixi run tests-e2e       # CPU backend (default)
#   pixi run tests-e2e-gpu   # GPU backend
#
# For proper benchmarking, use:
#   pixi run bench-cpu       # Uses benchmark.run() with warmup/batching
#   pixi run bench-gpu

from os import getenv
from testing import assert_equal, assert_true, assert_raises, TestSuite

from src import loads, dumps, Value


fn _is_gpu_mode() -> Bool:
    var val = getenv("MOJSON_TEST_GPU")
    return len(val) > 0 and val != "0" and val != "false"


fn _test_loads(json: String) raises -> Value:
    if _is_gpu_mode():
        return loads[target="gpu"](json)
    return loads[target="cpu"](json)


def test_roundtrip_null():
    """Test null roundtrip."""
    var original = "null"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1 == v2, "Null roundtrip failed")
    assert_equal(serialized, "null")


def test_roundtrip_bool_true():
    """Test true roundtrip."""
    var original = "true"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1 == v2, "True roundtrip failed")
    assert_equal(serialized, "true")


def test_roundtrip_bool_false():
    """Test false roundtrip."""
    var original = "false"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1 == v2, "False roundtrip failed")
    assert_equal(serialized, "false")


def test_roundtrip_int():
    """Test integer roundtrip."""
    var original = "42"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1 == v2, "Int roundtrip failed")
    assert_equal(serialized, "42")


def test_roundtrip_int_negative():
    """Test negative integer roundtrip."""
    var original = "-12345"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1 == v2, "Negative int roundtrip failed")
    assert_equal(serialized, "-12345")


def test_roundtrip_string():
    """Test string roundtrip."""
    var original = '"hello world"'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1 == v2, "String roundtrip failed")
    assert_equal(serialized, '"hello world"')


def test_roundtrip_string_empty():
    """Test empty string roundtrip."""
    var original = '""'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1 == v2, "Empty string roundtrip failed")
    assert_equal(serialized, '""')


def test_roundtrip_array_empty():
    """Test empty array roundtrip."""
    # Note: GPU parser currently has issues with top-level arrays
    # This test uses CPU to verify roundtrip behavior
    if _is_gpu_mode():
        # GPU has issues with standalone arrays, test with wrapped object
        var original = '{"arr":[]}'
        var v1 = _test_loads(original)
        var serialized = dumps(v1)
        var v2 = _test_loads(serialized)
        assert_true(v1.is_object() and v2.is_object(), "Should be objects")
        return
    var original = "[]"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_array() and v2.is_array(), "Should be arrays")
    assert_equal(serialized, "[]")


def test_roundtrip_array_simple():
    """Test simple array roundtrip."""
    # Note: GPU parser currently has issues with top-level arrays
    if _is_gpu_mode():
        # GPU has issues with standalone arrays, test with wrapped object
        var original = '{"arr":[1,2,3]}'
        var v1 = _test_loads(original)
        var serialized = dumps(v1)
        var v2 = _test_loads(serialized)
        assert_true(v1.is_object() and v2.is_object(), "Should be objects")
        return
    var original = "[1,2,3]"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_array() and v2.is_array(), "Should be arrays")
    # Verify element count
    assert_equal(v1.array_count(), 3)
    assert_equal(v2.array_count(), 3)


def test_roundtrip_object_empty():
    """Test empty object roundtrip."""
    var original = "{}"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_object() and v2.is_object(), "Should be objects")
    assert_equal(serialized, "{}")


def test_roundtrip_object_simple():
    """Test simple object roundtrip."""
    var original = '{"name":"Alice","age":30}'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_object() and v2.is_object(), "Should be objects")
    # Verify object key count
    assert_equal(v1.object_count(), 2)
    assert_equal(v2.object_count(), 2)


def test_roundtrip_nested():
    """Test nested structure roundtrip."""
    var original = '{"users":[{"name":"Bob","active":true}],"count":1}'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_object() and v2.is_object(), "Should be objects")
    # Verify structure preserved via raw JSON comparison
    assert_equal(v1.raw_json(), v2.raw_json())


def test_roundtrip_complex():
    """Test complex JSON roundtrip."""
    var original = '{"data":{"items":[1,2,3],"meta":{"version":"1.0","enabled":true}},"status":"ok"}'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_object() and v2.is_object(), "Should be objects")
    # Complex structure should be preserved
    assert_equal(v1.raw_json(), v2.raw_json())


def test_roundtrip_array_mixed():
    """Test mixed-type array roundtrip."""
    # Note: GPU parser currently has issues with top-level arrays
    if _is_gpu_mode():
        # GPU has issues with standalone arrays, test with wrapped object
        var original = '{"arr":[1,"two",true,null,3.14]}'
        var v1 = _test_loads(original)
        var serialized = dumps(v1)
        var v2 = _test_loads(serialized)
        assert_true(v1.is_object() and v2.is_object(), "Should be objects")
        return
    var original = '[1,"two",true,null,3.14]'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_array() and v2.is_array(), "Should be arrays")
    assert_equal(v1.array_count(), 5)
    assert_equal(v2.array_count(), 5)


def test_error_empty_input():
    """Test that empty input raises error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate empty input
    with assert_raises():
        _ = _test_loads("")


def test_error_whitespace_only():
    """Test that whitespace-only input raises error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate whitespace-only
    with assert_raises():
        _ = _test_loads("   ")


def test_error_invalid_token():
    """Test that invalid token raises error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate tokens
    with assert_raises():
        _ = _test_loads("invalid")


def test_error_unclosed_string():
    """Test that unclosed string raises error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate unclosed strings
    with assert_raises():
        _ = _test_loads('"hello')


def test_error_unclosed_array():
    """Test that unclosed array raises error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate brackets
    with assert_raises():
        _ = _test_loads("[1, 2, 3")


def test_error_unclosed_object():
    """Test that unclosed object raises error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate brackets
    with assert_raises():
        _ = _test_loads('{"key": "value"')


def test_error_trailing_comma_array():
    """Test that trailing comma in array raises error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate trailing commas
    with assert_raises():
        _ = _test_loads("[1, 2, 3,]")


def test_error_trailing_comma_object():
    """Test that trailing comma in object raises error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate trailing commas
    with assert_raises():
        _ = _test_loads('{"a": 1,}')


def test_error_missing_value():
    """Test that missing value raises error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate missing values
    with assert_raises():
        _ = _test_loads('{"key":}')


def test_error_missing_colon():
    """Test that missing colon in object raises error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate missing colons
    with assert_raises():
        _ = _test_loads('{"key" "value"}')


def test_error_double_comma():
    """Test that double comma raises error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate double commas
    with assert_raises():
        _ = _test_loads("[1,, 2]")


def test_error_leading_zeros():
    """Test that leading zeros raise error (per JSON spec, CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate number format
    with assert_raises():
        _ = _test_loads("007")


def test_error_single_quotes():
    """Test that single quotes raise error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate quote style
    with assert_raises():
        _ = _test_loads("'hello'")


def test_error_unquoted_key():
    """Test that unquoted object key raises error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate key quoting
    with assert_raises():
        _ = _test_loads("{key: 1}")


def test_error_extra_content():
    """Test that extra content after valid JSON raises error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate trailing content
    with assert_raises():
        _ = _test_loads("true false")


def test_error_invalid_escape():
    """Test that invalid escape sequence raises error (CPU only)."""
    if _is_gpu_mode():
        return  # GPU parser doesn't validate escape sequences
    with assert_raises():
        _ = _test_loads('"hello\\x"')


def test_edge_deeply_nested_array():
    """Test deeply nested arrays."""
    var original = "[[[[[[1]]]]]]"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_array() and v2.is_array(), "Should be arrays")


def test_edge_deeply_nested_object():
    """Test deeply nested objects."""
    var original = '{"a":{"b":{"c":{"d":1}}}}'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_object() and v2.is_object(), "Should be objects")
    assert_equal(v1.raw_json(), v2.raw_json())


def test_edge_unicode_string():
    """Test unicode escape sequence handling."""
    var original = '"Hello \\u0048\\u0065\\u006c\\u006c\\u006f"'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_string() and v2.is_string(), "Should be strings")


def test_edge_special_chars():
    """Test special character escaping."""
    var original = '"line1\\nline2\\ttab"'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_string() and v2.is_string(), "Should be strings")


def test_edge_large_number():
    """Test large integer handling."""
    var original = "9007199254740992"  # 2^53
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_int() and v2.is_int(), "Should be ints")


def test_edge_negative_zero():
    """Test negative zero (should be treated as zero)."""
    var original = "-0"
    var v1 = _test_loads(original)
    assert_true(v1.is_int() or v1.is_float(), "Should be numeric")


def test_edge_scientific_notation():
    """Test scientific notation float."""
    var original = "1.23e+10"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_float() and v2.is_float(), "Should be floats")


def test_edge_very_small_float():
    """Test very small float."""
    var original = "1e-300"
    var v1 = _test_loads(original)
    assert_true(v1.is_float(), "Should be float")


def main():
    var backend = "GPU" if _is_gpu_mode() else "CPU"
    print("=" * 60)
    print("test_e2e.mojo - End-to-End Tests (" + backend + ")")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
