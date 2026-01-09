# Tests for mojson/value.mojo

from testing import assert_equal, assert_true, assert_false, TestSuite

from src import Value, Null


def test_null_creation():
    """Test null value creation."""
    var v = Value(None)
    assert_true(v.is_null(), "Should be null")
    assert_equal(String(v), "null")


def test_null_from_null_type():
    """Test null from Null type."""
    var v = Value(Null())
    assert_true(v.is_null(), "Should be null")


def test_bool_true():
    """Test boolean true."""
    var v = Value(True)
    assert_true(v.is_bool(), "Should be bool")
    assert_true(v.bool_value(), "Should be true")
    assert_equal(String(v), "true")


def test_bool_false():
    """Test boolean false."""
    var v = Value(False)
    assert_true(v.is_bool(), "Should be bool")
    assert_false(v.bool_value(), "Should be false")
    assert_equal(String(v), "false")


def test_int_positive():
    """Test positive integer."""
    var v = Value(42)
    assert_true(v.is_int(), "Should be int")
    assert_true(v.is_number(), "Should be number")
    assert_equal(Int(v.int_value()), 42)


def test_int_negative():
    """Test negative integer."""
    var v = Value(-123)
    assert_true(v.is_int(), "Should be int")
    assert_equal(Int(v.int_value()), -123)


def test_int_zero():
    """Test zero."""
    var v = Value(0)
    assert_true(v.is_int(), "Should be int")
    assert_equal(Int(v.int_value()), 0)


def test_float():
    """Test float value."""
    var v = Value(3.14)
    assert_true(v.is_float(), "Should be float")
    assert_true(v.is_number(), "Should be number")


def test_string():
    """Test string value."""
    var v = Value("hello")
    assert_true(v.is_string(), "Should be string")
    assert_equal(v.string_value(), "hello")


def test_string_empty():
    """Test empty string."""
    var v = Value("")
    assert_true(v.is_string(), "Should be string")
    assert_equal(v.string_value(), "")


def test_equality_null():
    """Test null equality."""
    var a = Value(None)
    var b = Value(None)
    assert_true(a == b, "Nulls should be equal")


def test_equality_bool():
    """Test bool equality."""
    var a = Value(True)
    var b = Value(True)
    assert_true(a == b, "Bools should be equal")


def test_equality_int():
    """Test int equality."""
    var a = Value(42)
    var b = Value(42)
    assert_true(a == b, "Ints should be equal")


def test_equality_string():
    """Test string equality."""
    var a = Value("hello")
    var b = Value("hello")
    assert_true(a == b, "Strings should be equal")


def test_inequality():
    """Test inequality."""
    var a = Value(1)
    var b = Value(2)
    assert_true(a != b, "Different values should not be equal")


def test_type_mismatch():
    """Test type mismatch."""
    var a = Value(1)
    var b = Value("1")
    assert_true(a != b, "Different types should not be equal")


def main():
    print("=" * 60)
    print("test_value.mojo")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
