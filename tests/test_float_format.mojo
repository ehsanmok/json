"""Tests for shortest round-trip float writing.

The property that matters is round-trip: a value written by this
library must read back as that value, for every finite double. It is
checked against the platform's correctly rounded parser over hundreds
of thousands of random bit patterns, not against any Mojo formatter --
`String(Float64)` fails this property itself, which is why the writer
no longer uses it.

Layout is checked separately, because "reads back correctly" does not
mean "reads well": `0.1` must not print as `1e-01`.
"""

from std.collections import List
from std.ffi import external_call
from std.memory import bitcast
from std.random import random_float64, random_ui64
from std.testing import assert_equal, assert_true

from json.writer import JsonWriter


def _render(v: Float64) raises -> String:
    var w = JsonWriter(capacity=64)
    w.write_float(v)
    return w^.finish_string()


def _render_f32(v: Float32) raises -> String:
    var w = JsonWriter(capacity=64)
    w.write_float(Float64(v))
    return w^.finish_string()


def _reference(var text: String) -> Float64:
    """A correctly rounded reading of `text`."""
    var c_str = text.as_c_string_span()
    return external_call["strtod", Float64](c_str.ptr(), Int(0))


# ===================================================================
# Round-trip
# ===================================================================


def test_round_trips_over_the_double_range() raises:
    """Every finite double reads back as itself.

    Random bit patterns cover subnormals, values near the exponent
    limits, and the long-significand cases that are exactly where the
    stdlib formatter goes wrong.
    """
    var failures = 0
    var checked = 0
    for _ in range(300000):
        var bits = random_ui64(0, UInt64.MAX)
        if (bits >> 52) & 0x7FF == 0x7FF:
            continue
        var value = bitcast[DType.float64](bits)
        checked += 1
        var text = _render(value)
        if _reference(text) != value:
            failures += 1
            if failures < 4:
                print("    mismatch:", text, "for bits", hex(bits))
    assert_equal(failures, 0)
    assert_true(checked > 250000)
    print("  test_round_trips_over_the_double_range passed")


def test_round_trips_on_ordinary_decimals() raises:
    """The same, for the magnitudes documents actually hold."""
    var failures = 0
    for _ in range(200000):
        var value = (random_float64() - 0.5) * 20000.0
        var text = _render(value)
        if _reference(text) != value:
            failures += 1
    assert_equal(failures, 0)
    print("  test_round_trips_on_ordinary_decimals passed")


def test_round_trips_the_known_stdlib_failures() raises:
    """Values the stdlib formatter gets wrong.

    `String(0x43903c0e61516cab)` prints `2.924564535875448e+17`, which
    denotes the neighbouring double. Pinned here so a future switch
    back to a stdlib formatter fails loudly.
    """
    var failing: List[UInt64] = [
        0x43903C0E61516CAB,
        0xC357C2C9A20F3239,
        0xC37306F25AB7B39F,
        0xC35DEB21717B8673,
        0xC356136FA2078661,
    ]
    for i in range(len(failing)):
        var value = bitcast[DType.float64](failing[i])
        assert_equal(_reference(_render(value)), value)
    print("  test_round_trips_the_known_stdlib_failures passed")


# ===================================================================
# Layout
# ===================================================================


def test_layout_fixed_notation() raises:
    assert_equal(_render(1.0), "1.0")
    assert_equal(_render(0.0), "0.0")
    assert_equal(_render(-0.0), "-0.0")
    assert_equal(_render(100.0), "100.0")
    assert_equal(_render(-2.5), "-2.5")
    assert_equal(_render(0.1), "0.1")
    assert_equal(_render(0.3), "0.3")
    assert_equal(_render(0.0001), "0.0001")
    assert_equal(_render(123456789.12345679), "123456789.12345679")
    assert_equal(_render(1e15), "1000000000000000.0")
    print("  test_layout_fixed_notation passed")


def test_layout_scientific_notation() raises:
    """Two exponent digits minimum, always signed."""
    assert_equal(_render(1e16), "1e+16")
    assert_equal(_render(1e21), "1e+21")
    assert_equal(_render(1e-5), "1e-05")
    assert_equal(_render(1.5e-7), "1.5e-07")
    assert_equal(_render(1e100), "1e+100")
    assert_equal(_render(1e-100), "1e-100")
    assert_equal(_render(5e-324), "5e-324")
    assert_equal(_render(1.7976931348623157e308), "1.7976931348623157e+308")
    assert_equal(_render(2.2250738585072014e-308), "2.2250738585072014e-308")
    print("  test_layout_scientific_notation passed")


def test_integral_floats_keep_a_fraction() raises:
    """`1.0` must not print as `1`.

    A float that printed as an integer would read back as one, so a
    round trip through text would change the document's shape even
    though every number in it still had the same value.
    """
    assert_equal(_render(1.0), "1.0")
    assert_equal(_render(42.0), "42.0")
    assert_equal(_render(-7.0), "-7.0")
    print("  test_integral_floats_keep_a_fraction passed")


def test_non_finite_writes_null() raises:
    """JSON has no spelling for infinity or NaN.

    Emitting `inf` produced a document nothing could read back. `null`
    is at least valid; a caller that would rather be told can set the
    serializer to refuse.
    """
    var infinity = 1.0 / 0.0
    assert_equal(_render(infinity), "null")
    assert_equal(_render(-infinity), "null")
    assert_equal(_render(infinity - infinity), "null")
    print("  test_non_finite_writes_null passed")


# ===================================================================
# Integers
# ===================================================================


def test_integer_boundaries() raises:
    var w = JsonWriter(capacity=64)
    w.write_int(Int64.MIN)
    assert_equal(w^.finish_string(), "-9223372036854775808")

    var w2 = JsonWriter(capacity=64)
    w2.write_int(Int64.MAX)
    assert_equal(w2^.finish_string(), "9223372036854775807")

    var w3 = JsonWriter(capacity=64)
    w3.write_uint(UInt64.MAX)
    assert_equal(w3^.finish_string(), "18446744073709551615")

    var w4 = JsonWriter(capacity=64)
    w4.write_uint(0)
    assert_equal(w4^.finish_string(), "0")
    print("  test_integer_boundaries passed")


def test_reset_reuses_the_buffer() raises:
    var w = JsonWriter(capacity=64)
    w.write_float(1.5)
    w.reset()
    w.write_float(2.5)
    assert_equal(w^.finish_string(), "2.5")
    print("  test_reset_reuses_the_buffer passed")


def main() raises:
    print("Round-trip:")
    test_round_trips_over_the_double_range()
    test_round_trips_on_ordinary_decimals()
    test_round_trips_the_known_stdlib_failures()
    print()

    print("Layout:")
    test_layout_fixed_notation()
    test_layout_scientific_notation()
    test_integral_floats_keep_a_fraction()
    test_non_finite_writes_null()
    print()

    print("Integers:")
    test_integer_boundaries()
    test_reset_reuses_the_buffer()
    print()

    print("All float formatting tests passed!")
