# Tests for CPU parsing (loads)

from std.testing import assert_equal, assert_true, assert_false, TestSuite

from json import loads


# =============================================================================
# Primitive Parsing
# =============================================================================


def test_loads_null() raises:
    """Test loading null."""
    var v = loads("null")
    assert_true(v.is_null(), "Should load null")


def test_loads_true() raises:
    """Test loading true."""
    var v = loads("true")
    assert_true(v.is_bool() and v.bool_value(), "Should load true")


def test_loads_false() raises:
    """Test loading false."""
    var v = loads("false")
    assert_true(v.is_bool() and not v.bool_value(), "Should load false")


def test_loads_int_positive() raises:
    """Test loading positive integer."""
    var v = loads("42")
    assert_true(v.is_int(), "Should load int")
    assert_equal(Int(v.int_value()), 42)


def test_loads_int_negative() raises:
    """Test loading negative integer."""
    var v = loads("-123")
    assert_equal(Int(v.int_value()), -123)


def test_loads_int_zero() raises:
    """Test loading zero."""
    var v = loads("0")
    assert_equal(Int(v.int_value()), 0)


def test_loads_float() raises:
    """Test loading float."""
    var v = loads("3.14")
    assert_true(v.is_float(), "Should load float")


def test_loads_float_scientific() raises:
    """Test loading scientific notation."""
    var v = loads("1.5e10")
    assert_true(v.is_float(), "Should load scientific")


def test_loads_float_negative_exp() raises:
    """Test loading negative exponent."""
    var v = loads("1e-5")
    assert_true(v.is_float(), "Should load negative exp")


# =============================================================================
# String Parsing
# =============================================================================


def test_loads_string() raises:
    """Test loading simple string."""
    var v = loads('"hello world"')
    assert_true(v.is_string(), "Should load string")
    assert_equal(v.string_value(), "hello world")


def test_loads_string_empty() raises:
    """Test loading empty string."""
    var v = loads('""')
    assert_equal(v.string_value(), "")


def test_loads_string_escape_newline() raises:
    """Test loading escaped newline."""
    var v = loads('"hello\\nworld"')
    assert_equal(v.string_value(), "hello\nworld")


def test_loads_string_escape_tab() raises:
    """Test loading escaped tab."""
    var v = loads('"hello\\tworld"')
    assert_equal(v.string_value(), "hello\tworld")


def test_loads_string_escape_quote() raises:
    """Test loading escaped quote."""
    var v = loads('"say \\"hello\\""')
    assert_equal(v.string_value(), 'say "hello"')


def test_loads_string_escape_backslash() raises:
    """Test loading escaped backslash."""
    var v = loads('"path\\\\file"')
    assert_equal(v.string_value(), "path\\file")


# =============================================================================
# Array Parsing
# =============================================================================


def test_loads_array_empty() raises:
    """Test loading empty array."""
    var v = loads("[]")
    assert_true(v.is_array(), "Should load array")


def test_loads_array_ints() raises:
    """Test loading array of ints."""
    var v = loads("[1, 2, 3]")
    assert_true(v.is_array(), "Should load array")


def test_loads_array_mixed() raises:
    """Test loading mixed array."""
    var v = loads('[1, "two", true, null]')
    assert_true(v.is_array(), "Should load mixed array")


def test_loads_array_nested() raises:
    """Test loading nested array."""
    var v = loads("[[1, 2], [3, 4]]")
    assert_true(v.is_array(), "Should load nested array")


# =============================================================================
# Object Parsing
# =============================================================================


def test_loads_object_empty() raises:
    """Test loading empty object."""
    var v = loads("{}")
    assert_true(v.is_object(), "Should load object")


def test_loads_object_simple() raises:
    """Test loading simple object."""
    var v = loads('{"name": "Alice", "age": 30}')
    assert_true(v.is_object(), "Should load object")


def test_loads_object_nested() raises:
    """Test loading nested object."""
    var v = loads('{"user": {"name": "Bob", "scores": [85, 90, 95]}}')
    assert_true(v.is_object(), "Should load nested")


# =============================================================================
# Whitespace Handling
# =============================================================================


def test_loads_whitespace_spaces() raises:
    """Test loading with spaces."""
    var v = loads('  {  "key"  :  "value"  }  ')
    assert_true(v.is_object(), "Should handle spaces")


def test_loads_whitespace_newlines() raises:
    """Test loading with newlines."""
    var v = loads('{\n"key":\n"value"\n}')
    assert_true(v.is_object(), "Should handle newlines")


def test_loads_whitespace_tabs() raises:
    """Test loading with tabs."""
    var v = loads('{\t"key":\t"value"\t}')
    assert_true(v.is_object(), "Should handle tabs")


# =============================================================================
# Error Handling
# =============================================================================


def test_error_empty() raises:
    """Test error on empty input."""
    var caught = False
    try:
        _ = loads("")
    except:
        caught = True
    assert_true(caught, "Should raise on empty")


def test_error_invalid() raises:
    """Test error on invalid JSON."""
    var caught = False
    try:
        _ = loads("invalid")
    except:
        caught = True
    assert_true(caught, "Should raise on invalid")


def test_error_unclosed_string() raises:
    """Test error on unclosed string."""
    var caught = False
    try:
        _ = loads('"unclosed')
    except:
        caught = True
    assert_true(caught, "Should raise on unclosed string")


def test_error_unclosed_array() raises:
    """Test error on unclosed array."""
    var caught = False
    try:
        _ = loads("[1, 2")
    except:
        caught = True
    assert_true(caught, "Should raise on unclosed array")


def test_error_unclosed_object() raises:
    """Test error on unclosed object."""
    var caught = False
    try:
        _ = loads('{"key": "value"')
    except:
        caught = True
    assert_true(caught, "Should raise on unclosed object")


def test_error_trailing_comma_array() raises:
    """Test error on trailing comma in array."""
    var caught = False
    try:
        _ = loads("[1, 2,]")
    except:
        caught = True
    assert_true(caught, "Should raise on trailing comma")


def test_error_trailing_comma_object() raises:
    """Test error on trailing comma in object."""
    var caught = False
    try:
        _ = loads('{"a": 1,}')
    except:
        caught = True
    assert_true(caught, "Should raise on trailing comma")


def test_error_message_has_context() raises:
    """Test that error messages include context info."""
    var caught = False
    var msg = String()
    try:
        _ = loads('{\n  "key": \n}')
    except e:
        caught = True
        msg = String(e)
    assert_true(caught, "Should raise error")
    # Error should have some useful info
    assert_true(msg.byte_length() > 10, "Error should have descriptive message")


def test_error_message_multiline() raises:
    """Test error on multiline JSON."""
    var caught = False
    var msg = String()
    var json = '{\n  "name": "test",\n  "value": \n}'
    try:
        _ = loads(json)
    except e:
        caught = True
        msg = String(e)
    assert_true(caught, "Should raise error")
    # Just verify it returns something useful
    assert_true(msg.byte_length() > 0, "Should have error message")


# =============================================================================
# Unicode Escape Tests
# =============================================================================


def test_unicode_escape_basic() raises:
    """Test basic unicode escape \\u0041 = 'A'."""
    var v = loads('"\\u0041"')
    assert_equal(v.string_value(), "A")


def test_unicode_escape_euro() raises:
    """Test unicode escape for euro sign \\u20AC."""
    var v = loads('"\\u20AC"')
    assert_equal(v.string_value(), "€")


def test_unicode_escape_mixed() raises:
    """Test mixed unicode and regular text."""
    var v = loads('"Hello \\u0057orld"')  # \u0057 = 'W'
    assert_equal(v.string_value(), "Hello World")


def test_unicode_escape_null_char() raises:
    """Test unicode null character \\u0000."""
    var v = loads('"a\\u0000b"')
    # Should have 3 characters: 'a', null, 'b'
    var s = v.string_value()
    assert_equal(len(s.as_bytes()), 3)


def test_unicode_surrogate_pair() raises:
    """Test surrogate pair for emoji (U+1F600 = 😀)."""
    # U+1F600 = D83D DE00 (surrogate pair)
    var v = loads('"\\uD83D\\uDE00"')
    var s = v.string_value()
    # 😀 in UTF-8 is 4 bytes: F0 9F 98 80
    assert_equal(len(s.as_bytes()), 4)


def test_unicode_in_object() raises:
    """Test unicode escapes in object values."""
    var v = loads('{"name": "\\u0041lice", "emoji": "\\u2764"}')
    assert_true(v.is_object(), "Should be object")


def main() raises:
    print("=" * 60)
    print("test_parser.mojo - CPU loads() tests")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
