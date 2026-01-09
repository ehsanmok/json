# Tests for mojson/serialize.mojo

from testing import assert_equal, assert_true, TestSuite

from src import Value, Null, dumps


def test_serialize_null():
    """Test serializing null."""
    var v = Value(None)
    assert_equal(dumps(v), "null")


def test_serialize_true():
    """Test serializing true."""
    var v = Value(True)
    assert_equal(dumps(v), "true")


def test_serialize_false():
    """Test serializing false."""
    var v = Value(False)
    assert_equal(dumps(v), "false")


def test_serialize_int_positive():
    """Test serializing positive int."""
    var v = Value(42)
    assert_equal(dumps(v), "42")


def test_serialize_int_negative():
    """Test serializing negative int."""
    var v = Value(-123)
    assert_equal(dumps(v), "-123")


def test_serialize_int_zero():
    """Test serializing zero."""
    var v = Value(0)
    assert_equal(dumps(v), "0")


def test_serialize_string():
    """Test serializing string."""
    var v = Value("hello")
    assert_equal(dumps(v), '"hello"')


def test_serialize_string_empty():
    """Test serializing empty string."""
    var v = Value("")
    assert_equal(dumps(v), '""')


def main():
    print("=" * 60)
    print("test_serialize.mojo")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
