# End-to-end tests for json
# Tests JSON roundtrip (loads -> dumps -> loads) and error handling
#
# Usage:
#   pixi run tests-e2e       # CPU backend (default)
#   pixi run tests-e2e-gpu   # GPU backend
#
# For proper benchmarking, use:
#   pixi run bench-cpu       # Uses benchmark.run() with warmup/batching
#   pixi run bench-gpu

from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from json import loads, dumps, Value


def _test_loads(json: String) raises -> Value:
    """The loader under test.

    This used to switch to the GPU backend when `MOJSON_TEST_GPU` was
    set. The GPU entry point now lives in `json.gpu`, which this file
    deliberately does not import: keeping the CPU suite free of that
    import is what keeps `max-core` out of the default environment.
    `packages`-free equivalent coverage for the GPU path is in
    `tests/test_gpu.mojo`.
    """
    return loads[target="cpu"](json)


def test_roundtrip_null() raises:
    """Test null roundtrip."""
    var original = "null"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1 == v2, "Null roundtrip failed")
    assert_equal(serialized, "null")


def test_roundtrip_bool_true() raises:
    """Test true roundtrip."""
    var original = "true"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1 == v2, "True roundtrip failed")
    assert_equal(serialized, "true")


def test_roundtrip_bool_false() raises:
    """Test false roundtrip."""
    var original = "false"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1 == v2, "False roundtrip failed")
    assert_equal(serialized, "false")


def test_roundtrip_int() raises:
    """Test integer roundtrip."""
    var original = "42"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1 == v2, "Int roundtrip failed")
    assert_equal(serialized, "42")


def test_roundtrip_int_negative() raises:
    """Test negative integer roundtrip."""
    var original = "-12345"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1 == v2, "Negative int roundtrip failed")
    assert_equal(serialized, "-12345")


def test_roundtrip_string() raises:
    """Test string roundtrip."""
    var original = '"hello world"'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1 == v2, "String roundtrip failed")
    assert_equal(serialized, '"hello world"')


def test_roundtrip_string_empty() raises:
    """Test empty string roundtrip."""
    var original = '""'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1 == v2, "Empty string roundtrip failed")
    assert_equal(serialized, '""')


def test_roundtrip_array_empty() raises:
    """Test empty array roundtrip."""
    # Note: GPU parser currently has issues with top-level arrays
    # This test uses CPU to verify roundtrip behavior
    var original = "[]"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_array() and v2.is_array(), "Should be arrays")
    assert_equal(serialized, "[]")


def test_roundtrip_array_simple() raises:
    """Test simple array roundtrip."""
    var original = "[1,2,3]"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_array() and v2.is_array(), "Should be arrays")
    # Verify element count
    assert_equal(v1.array_count(), 3)
    assert_equal(v2.array_count(), 3)


def test_roundtrip_object_empty() raises:
    """Test empty object roundtrip."""
    var original = "{}"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_object() and v2.is_object(), "Should be objects")
    assert_equal(serialized, "{}")


def test_roundtrip_object_simple() raises:
    """Test simple object roundtrip."""
    var original = '{"name":"Alice","age":30}'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_object() and v2.is_object(), "Should be objects")
    # Verify object key count
    assert_equal(v1.object_count(), 2)
    assert_equal(v2.object_count(), 2)


def test_roundtrip_nested() raises:
    """Test nested structure roundtrip."""
    var original = '{"users":[{"name":"Bob","active":true}],"count":1}'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_object() and v2.is_object(), "Should be objects")
    # Verify structure preserved via raw JSON comparison
    assert_equal(v1.raw_json(), v2.raw_json())


def test_roundtrip_complex() raises:
    """Test complex JSON roundtrip."""
    var original = '{"data":{"items":[1,2,3],"meta":{"version":"1.0","enabled":true}},"status":"ok"}'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_object() and v2.is_object(), "Should be objects")
    # Complex structure should be preserved
    assert_equal(v1.raw_json(), v2.raw_json())


def test_roundtrip_array_mixed() raises:
    """Test mixed-type array roundtrip."""
    var original = '[1,"two",true,null,3.14]'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_array() and v2.is_array(), "Should be arrays")
    assert_equal(v1.array_count(), 5)
    assert_equal(v2.array_count(), 5)


def test_error_empty_input() raises:
    """Test that empty input raises error (CPU only)."""
    with assert_raises():
        _ = _test_loads("")


def test_error_whitespace_only() raises:
    """Test that whitespace-only input raises error (CPU only)."""
    with assert_raises():
        _ = _test_loads("   ")


def test_error_invalid_token() raises:
    """Test that invalid token raises error (CPU only)."""
    with assert_raises():
        _ = _test_loads("invalid")


def test_error_unclosed_string() raises:
    """Test that unclosed string raises error (CPU only)."""
    with assert_raises():
        _ = _test_loads('"hello')


def test_error_unclosed_array() raises:
    """Test that unclosed array raises error (CPU only)."""
    with assert_raises():
        _ = _test_loads("[1, 2, 3")


def test_error_unclosed_object() raises:
    """Test that unclosed object raises error (CPU only)."""
    with assert_raises():
        _ = _test_loads('{"key": "value"')


def test_error_trailing_comma_array() raises:
    """Test that trailing comma in array raises error (CPU only)."""
    with assert_raises():
        _ = _test_loads("[1, 2, 3,]")


def test_error_trailing_comma_object() raises:
    """Test that trailing comma in object raises error (CPU only)."""
    with assert_raises():
        _ = _test_loads('{"a": 1,}')


def test_error_missing_value() raises:
    """Test that missing value raises error (CPU only)."""
    with assert_raises():
        _ = _test_loads('{"key":}')


def test_error_missing_colon() raises:
    """Test that missing colon in object raises error (CPU only)."""
    with assert_raises():
        _ = _test_loads('{"key" "value"}')


def test_error_double_comma() raises:
    """Test that double comma raises error (CPU only)."""
    with assert_raises():
        _ = _test_loads("[1,, 2]")


def test_error_leading_zeros() raises:
    """Test that leading zeros raise error (per JSON spec, CPU only)."""
    with assert_raises():
        _ = _test_loads("007")


def test_error_single_quotes() raises:
    """Test that single quotes raise error (CPU only)."""
    with assert_raises():
        _ = _test_loads("'hello'")


def test_error_unquoted_key() raises:
    """Test that unquoted object key raises error (CPU only)."""
    with assert_raises():
        _ = _test_loads("{key: 1}")


def test_error_extra_content() raises:
    """Test that extra content after valid JSON raises error (CPU only)."""
    with assert_raises():
        _ = _test_loads("true false")


def test_error_invalid_escape() raises:
    """Test that invalid escape sequence raises error (CPU only)."""
    with assert_raises():
        _ = _test_loads('"hello\\x"')


def test_edge_deeply_nested_array() raises:
    """Test deeply nested arrays."""
    var original = "[[[[[[1]]]]]]"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_array() and v2.is_array(), "Should be arrays")


def test_edge_deeply_nested_object() raises:
    """Test deeply nested objects."""
    var original = '{"a":{"b":{"c":{"d":1}}}}'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_object() and v2.is_object(), "Should be objects")
    assert_equal(v1.raw_json(), v2.raw_json())


def test_edge_unicode_string() raises:
    """Test unicode escape sequence handling."""
    var original = '"Hello \\u0048\\u0065\\u006c\\u006c\\u006f"'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_string() and v2.is_string(), "Should be strings")


def test_edge_special_chars() raises:
    """Test special character escaping."""
    var original = '"line1\\nline2\\ttab"'
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_string() and v2.is_string(), "Should be strings")


def test_edge_large_number() raises:
    """Test large integer handling."""
    var original = "9007199254740992"  # 2^53
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_int() and v2.is_int(), "Should be ints")


def test_edge_negative_zero() raises:
    """Test negative zero (should be treated as zero)."""
    var original = "-0"
    var v1 = _test_loads(original)
    assert_true(v1.is_int() or v1.is_float(), "Should be numeric")


def test_edge_scientific_notation() raises:
    """Test scientific notation float."""
    var original = "1.23e+10"
    var v1 = _test_loads(original)
    var serialized = dumps(v1)
    var v2 = _test_loads(serialized)
    assert_true(v1.is_float() and v2.is_float(), "Should be floats")


def test_edge_very_small_float() raises:
    """Test very small float."""
    var original = "1e-300"
    var v1 = _test_loads(original)
    assert_true(v1.is_float(), "Should be float")


def main() raises:
    print("=" * 60)
    print("test_e2e.mojo - End-to-End Tests (CPU)")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
