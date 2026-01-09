# GPU loading tests
# Tests: loads[target="gpu"](...)

from testing import assert_equal, assert_true, TestSuite

from src import loads, dumps, Value, Null


# =============================================================================
# GPU loads Tests
# =============================================================================


fn test_loads_gpu_simple_object() raises:
    """Test GPU loads with simple object."""
    var v = loads[target="gpu"]('{"name": "Alice"}')
    assert_true(v.is_object(), "GPU should return object")


fn test_loads_gpu_simple_array() raises:
    """Test GPU loads with simple array."""
    var v = loads[target="gpu"]("[1, 2, 3, 4, 5]")
    assert_true(v.is_array(), "GPU should return array")


fn test_loads_gpu_nested() raises:
    """Test GPU loads with nested structure."""
    var v = loads[target="gpu"]('{"data": {"nested": [1, 2, 3]}}')
    assert_true(v.is_object(), "GPU should handle nested structures")


fn test_loads_gpu_string() raises:
    """Test GPU loads with string."""
    var v = loads[target="gpu"]('"hello world"')
    assert_true(v.is_string(), "GPU should return string")


fn test_loads_gpu_number() raises:
    """Test GPU loads with number."""
    var v = loads[target="gpu"]("12345")
    assert_true(v.is_int() or v.is_float(), "GPU should return number")


fn test_loads_gpu_bool() raises:
    """Test GPU loads with boolean."""
    var v = loads[target="gpu"]("true")
    assert_true(v.is_bool(), "GPU should return bool")


fn test_loads_gpu_null() raises:
    """Test GPU loads with null."""
    var v = loads[target="gpu"]("null")
    assert_true(v.is_null(), "GPU should return null")


fn test_loads_gpu_bool_false() raises:
    """Test GPU loads with false."""
    var v = loads[target="gpu"]("false")
    assert_true(v.is_bool(), "GPU should return bool")
    assert_equal(v.bool_value(), False)


fn test_loads_gpu_negative_number() raises:
    """Test GPU loads with negative number."""
    var v = loads[target="gpu"]("-42")
    assert_true(v.is_int() or v.is_float(), "GPU should return number")


fn test_loads_gpu_float() raises:
    """Test GPU loads with float."""
    var v = loads[target="gpu"]("3.14159")
    assert_true(v.is_float(), "GPU should return float")


fn test_loads_gpu_empty_object() raises:
    """Test GPU loads with empty object."""
    var v = loads[target="gpu"]("{}")
    assert_true(v.is_object(), "GPU should return empty object")


fn test_loads_gpu_empty_array() raises:
    """Test GPU loads with empty array."""
    var v = loads[target="gpu"]("[]")
    assert_true(v.is_array(), "GPU should return empty array")


fn test_loads_gpu_array_of_objects() raises:
    """Test GPU loads with array of objects."""
    var v = loads[target="gpu"]('[{"a": 1}, {"b": 2}]')
    assert_true(v.is_array(), "GPU should return array")


fn test_loads_gpu_deeply_nested() raises:
    """Test GPU loads with deeply nested structure."""
    var v = loads[target="gpu"]('{"a": {"b": {"c": {"d": 1}}}}')
    assert_true(v.is_object(), "GPU should handle deep nesting")


# =============================================================================
# CPU/GPU Equivalence Tests
# =============================================================================


fn test_cpu_gpu_equivalence_object() raises:
    """Test CPU and GPU produce equivalent results for objects."""
    var json = '{"a": 1, "b": 2}'
    var cpu_result = loads[target="cpu"](json)
    var gpu_result = loads[target="gpu"](json)
    assert_equal(cpu_result.is_object(), gpu_result.is_object())


fn test_cpu_gpu_equivalence_array() raises:
    """Test CPU and GPU produce equivalent results for arrays."""
    var json = "[1, 2, 3, 4, 5]"
    var cpu_result = loads[target="cpu"](json)
    var gpu_result = loads[target="gpu"](json)
    assert_equal(cpu_result.is_array(), gpu_result.is_array())


fn test_cpu_gpu_equivalence_nested() raises:
    """Test CPU and GPU produce equivalent results for nested structures."""
    var json = '{"users": [{"name": "Alice"}, {"name": "Bob"}], "count": 2}'
    var cpu_result = loads[target="cpu"](json)
    var gpu_result = loads[target="gpu"](json)
    assert_equal(cpu_result.is_object(), gpu_result.is_object())


fn test_gpu_dumps_roundtrip() raises:
    """Test GPU loads then dumps roundtrip."""
    var json = '{"test": "value"}'
    var v = loads[target="gpu"](json)
    var output = dumps(v)
    # Verify output is valid JSON by loading again
    var v2 = loads[target="cpu"](output)
    assert_true(v2.is_object(), "GPU roundtrip should produce valid object")


def main():
    print("=" * 60)
    print("test_gpu.mojo - GPU loads() tests")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
