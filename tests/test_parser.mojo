# Tests for CPU parsing (loads)

from testing import assert_equal, assert_true, assert_false, TestSuite

from src import loads


# =============================================================================
# Primitive Parsing
# =============================================================================


def test_loads_null():
    """Test loading null."""
    var v = loads("null")
    assert_true(v.is_null(), "Should load null")


def test_loads_true():
    """Test loading true."""
    var v = loads("true")
    assert_true(v.is_bool() and v.bool_value(), "Should load true")


def test_loads_false():
    """Test loading false."""
    var v = loads("false")
    assert_true(v.is_bool() and not v.bool_value(), "Should load false")


def test_loads_int_positive():
    """Test loading positive integer."""
    var v = loads("42")
    assert_true(v.is_int(), "Should load int")
    assert_equal(Int(v.int_value()), 42)


def test_loads_int_negative():
    """Test loading negative integer."""
    var v = loads("-123")
    assert_equal(Int(v.int_value()), -123)


def test_loads_int_zero():
    """Test loading zero."""
    var v = loads("0")
    assert_equal(Int(v.int_value()), 0)


def test_loads_float():
    """Test loading float."""
    var v = loads("3.14")
    assert_true(v.is_float(), "Should load float")


def test_loads_float_scientific():
    """Test loading scientific notation."""
    var v = loads("1.5e10")
    assert_true(v.is_float(), "Should load scientific")


def test_loads_float_negative_exp():
    """Test loading negative exponent."""
    var v = loads("1e-5")
    assert_true(v.is_float(), "Should load negative exp")


# =============================================================================
# String Parsing
# =============================================================================


def test_loads_string():
    """Test loading simple string."""
    var v = loads('"hello world"')
    assert_true(v.is_string(), "Should load string")
    assert_equal(v.string_value(), "hello world")


def test_loads_string_empty():
    """Test loading empty string."""
    var v = loads('""')
    assert_equal(v.string_value(), "")


def test_loads_string_escape_newline():
    """Test loading escaped newline."""
    var v = loads('"hello\\nworld"')
    assert_equal(v.string_value(), "hello\nworld")


def test_loads_string_escape_tab():
    """Test loading escaped tab."""
    var v = loads('"hello\\tworld"')
    assert_equal(v.string_value(), "hello\tworld")


def test_loads_string_escape_quote():
    """Test loading escaped quote."""
    var v = loads('"say \\"hello\\""')
    assert_equal(v.string_value(), 'say "hello"')


def test_loads_string_escape_backslash():
    """Test loading escaped backslash."""
    var v = loads('"path\\\\file"')
    assert_equal(v.string_value(), "path\\file")


# =============================================================================
# Array Parsing
# =============================================================================


def test_loads_array_empty():
    """Test loading empty array."""
    var v = loads("[]")
    assert_true(v.is_array(), "Should load array")


def test_loads_array_ints():
    """Test loading array of ints."""
    var v = loads("[1, 2, 3]")
    assert_true(v.is_array(), "Should load array")


def test_loads_array_mixed():
    """Test loading mixed array."""
    var v = loads('[1, "two", true, null]')
    assert_true(v.is_array(), "Should load mixed array")


def test_loads_array_nested():
    """Test loading nested array."""
    var v = loads("[[1, 2], [3, 4]]")
    assert_true(v.is_array(), "Should load nested array")


# =============================================================================
# Object Parsing
# =============================================================================


def test_loads_object_empty():
    """Test loading empty object."""
    var v = loads("{}")
    assert_true(v.is_object(), "Should load object")


def test_loads_object_simple():
    """Test loading simple object."""
    var v = loads('{"name": "Alice", "age": 30}')
    assert_true(v.is_object(), "Should load object")


def test_loads_object_nested():
    """Test loading nested object."""
    var v = loads('{"user": {"name": "Bob", "scores": [85, 90, 95]}}')
    assert_true(v.is_object(), "Should load nested")


# =============================================================================
# Whitespace Handling
# =============================================================================


def test_loads_whitespace_spaces():
    """Test loading with spaces."""
    var v = loads('  {  "key"  :  "value"  }  ')
    assert_true(v.is_object(), "Should handle spaces")


def test_loads_whitespace_newlines():
    """Test loading with newlines."""
    var v = loads('{\n"key":\n"value"\n}')
    assert_true(v.is_object(), "Should handle newlines")


def test_loads_whitespace_tabs():
    """Test loading with tabs."""
    var v = loads('{\t"key":\t"value"\t}')
    assert_true(v.is_object(), "Should handle tabs")


# =============================================================================
# Error Handling
# =============================================================================


def test_error_empty():
    """Test error on empty input."""
    var caught = False
    try:
        _ = loads("")
    except:
        caught = True
    assert_true(caught, "Should raise on empty")


def test_error_invalid():
    """Test error on invalid JSON."""
    var caught = False
    try:
        _ = loads("invalid")
    except:
        caught = True
    assert_true(caught, "Should raise on invalid")


def test_error_unclosed_string():
    """Test error on unclosed string."""
    var caught = False
    try:
        _ = loads('"unclosed')
    except:
        caught = True
    assert_true(caught, "Should raise on unclosed string")


def test_error_unclosed_array():
    """Test error on unclosed array."""
    var caught = False
    try:
        _ = loads("[1, 2")
    except:
        caught = True
    assert_true(caught, "Should raise on unclosed array")


def test_error_unclosed_object():
    """Test error on unclosed object."""
    var caught = False
    try:
        _ = loads('{"key": "value"')
    except:
        caught = True
    assert_true(caught, "Should raise on unclosed object")


def test_error_trailing_comma_array():
    """Test error on trailing comma in array."""
    var caught = False
    try:
        _ = loads("[1, 2,]")
    except:
        caught = True
    assert_true(caught, "Should raise on trailing comma")


def test_error_trailing_comma_object():
    """Test error on trailing comma in object."""
    var caught = False
    try:
        _ = loads('{"a": 1,}')
    except:
        caught = True
    assert_true(caught, "Should raise on trailing comma")


def main():
    print("=" * 60)
    print("test_parser.mojo - CPU loads() tests")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
