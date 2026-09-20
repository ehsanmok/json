"""Tests for the JSON number tokenizer.

Two properties matter here and they pull in opposite directions. The
scanner must reject everything outside RFC 8259 section 6, including
the near-misses that a permissive "consume digits and punctuation"
loop accepts. And it must convert what it does accept to the same
`Float64` the platform's correctly-rounded parser produces, which is
checked differentially against `atof` over random values rather than
against a handful of literals.
"""

from std.collections import List
from std.ffi import external_call
from std.memory import bitcast
from std.random import random_float64, random_ui64
from std.testing import assert_equal, assert_raises, assert_true

from json.cpu.number_parse import (
    NUM_ERR_GRAMMAR,
    NUM_ERR_LEADING_ZERO,
    NUM_FLOAT,
    NUM_INT,
    NUM_INVALID,
    NUM_UINT,
    parse_float_span,
    parse_int_checked,
    scan_number,
)


def _reference(var text: String) -> Float64:
    """A correctly rounded reading of `text`, for differential checks.

    The platform `strtod` is the reference because the two obvious
    in-language candidates are not: `atof` is one unit in the last
    place off for roughly one random double in a thousand, and
    `String(Float64)` does not always produce a representation that
    reads back as itself.
    """
    var c_str = text.as_c_string_span()
    return external_call["strtod", Float64](c_str.ptr(), Int(0))


def _scan(text: String) raises -> Tuple[UInt8, Int, Int64, UInt64, Float64]:
    var token = scan_number(text.as_bytes(), 0, text.byte_length())
    return (
        token.kind,
        token.end,
        token.int_value,
        token.uint_value,
        token.float_value,
    )


def _must_reject(text: String) raises:
    var token = scan_number(text.as_bytes(), 0, text.byte_length())
    assert_equal(token.kind, NUM_INVALID, "should have rejected: " + text)


def _int_of(text: String) raises -> Int64:
    var token = scan_number(text.as_bytes(), 0, text.byte_length())
    assert_equal(token.kind, NUM_INT, "not an Int64: " + text)
    return token.int_value


def _float_of(text: String) raises -> Float64:
    var token = scan_number(text.as_bytes(), 0, text.byte_length())
    assert_equal(token.kind, NUM_FLOAT, "not a float: " + text)
    return token.float_value


# ===================================================================
# Grammar
# ===================================================================


def test_rejects_malformed_numbers() raises:
    """Every near-miss the old consume-everything loop accepted."""
    var bad: List[String] = [
        "-",
        "-.",
        "-2.",
        "1.",
        ".5",
        "-.123",
        "+1",
        "1e",
        "1e+",
        "1e-",
        "0.e1",
        "2.e3",
        "2.e+3",
        "2.e-3",
        "1eE2",
        "0e+-1",
        "9.e+",
        "0.3e",
        "0.3e+",
        "1.0e",
        "e5",
        "E5",
        ".e1",
    ]
    for i in range(len(bad)):
        _must_reject(bad[i])
    print("  test_rejects_malformed_numbers passed")


def test_rejects_leading_zeros() raises:
    """`01` is two tokens to the grammar, not a number."""
    for text in ["01", "-01", "00", "-00", "012", "00.5"]:
        var token = scan_number(text.as_bytes(), 0, text.byte_length())
        assert_equal(token.kind, NUM_INVALID, text)
        assert_equal(token.err, NUM_ERR_LEADING_ZERO, text)
    # A single zero, and a zero with a fraction, are both fine.
    assert_equal(_int_of("0"), 0)
    assert_equal(_float_of("0.5"), 0.5)
    assert_equal(_float_of("0e0"), 0.0)
    print("  test_rejects_leading_zeros passed")


def test_stops_without_consuming() raises:
    """The scanner leaves the first byte it cannot use.

    This is what lets a caller say "expected ',' or ']'" for `[1+2]`
    instead of reporting a number error: the number `1` is valid and
    complete, and `+` is the caller's problem.
    """
    var r = _scan("1+2")
    assert_equal(r[0], NUM_INT)
    assert_equal(r[1], 1)
    assert_equal(r[2], 1)

    var s = _scan("12,34")
    assert_equal(s[1], 2)
    assert_equal(s[2], 12)

    var t = _scan("1.5e3]")
    assert_equal(t[0], NUM_FLOAT)
    assert_equal(t[1], 5)

    # `0_1` is a complete `0` followed by a byte the caller must
    # reject. The scanner reporting a number error here would name the
    # wrong thing.
    var u = _scan("0_1")
    assert_equal(u[0], NUM_INT)
    assert_equal(u[1], 1)
    print("  test_stops_without_consuming passed")


def test_accepts_exponent_spellings() raises:
    assert_equal(_float_of("1e2"), 100.0)
    assert_equal(_float_of("1E2"), 100.0)
    assert_equal(_float_of("1e+2"), 100.0)
    assert_equal(_float_of("1E-2"), 0.01)
    assert_equal(_float_of("0E0"), 0.0)
    assert_equal(_float_of("1.5E+3"), 1500.0)
    print("  test_accepts_exponent_spellings passed")


# ===================================================================
# Integer range
# ===================================================================


def test_int64_boundaries() raises:
    assert_equal(_int_of("9223372036854775807"), Int64.MAX)
    assert_equal(_int_of("-9223372036854775808"), Int64.MIN)
    assert_equal(_int_of("-0"), 0)
    assert_equal(_int_of("0"), 0)
    print("  test_int64_boundaries passed")


def test_above_int64_is_unsigned() raises:
    """A positive integer past `Int64.MAX` keeps every digit."""
    var a = _scan("9223372036854775808")
    assert_equal(a[0], NUM_UINT)
    assert_equal(a[3], UInt64(1) << 63)

    var b = _scan("18446744073709551615")
    assert_equal(b[0], NUM_UINT)
    assert_equal(b[3], UInt64.MAX)
    print("  test_above_int64_is_unsigned passed")


def test_beyond_uint64_becomes_float() raises:
    """Past `UInt64`, the nearest `Float64` -- never a wrapped integer.

    Wrapping is the one answer that is silently wrong:
    `18446744073709551616` used to read back as a small negative
    number with no error anywhere.
    """
    var a = _scan("18446744073709551616")
    assert_equal(a[0], NUM_FLOAT)
    assert_equal(a[4], 1.8446744073709552e19)

    var b = _scan("-9223372036854775809")
    assert_equal(b[0], NUM_FLOAT)
    assert_true(b[4] < -9.223372036854775e18)

    var c = _scan("123456789012345678901234567890")
    assert_equal(c[0], NUM_FLOAT)
    assert_equal(c[4], 1.2345678901234568e29)
    print("  test_beyond_uint64_becomes_float passed")


# ===================================================================
# Float conversion
# ===================================================================


def test_float_edges() raises:
    assert_equal(_float_of("0.1"), 0.1)
    assert_equal(_float_of("1e23"), 1e23)
    assert_equal(_float_of("5e-324"), 5e-324)
    assert_equal(_float_of("2.2250738585072014e-308"), 2.2250738585072014e-308)
    assert_equal(_float_of("1.7976931348623157e308"), 1.7976931348623157e308)
    assert_equal(_float_of("-0.0"), -0.0)
    assert_equal(_float_of("1e-400"), 0.0)
    print("  test_float_edges passed")


def test_long_mantissa_is_correctly_rounded() raises:
    """More significant digits than a `Float64` holds still converts.

    The stdlib string parser refuses these outright, which is why
    `y_number_double_close_to_zero` from the conformance corpus was
    being rejected as invalid JSON.
    """
    var tiny = String(
        "-0.000000000000000000000000000000000000000000000000000000000"
    )
    tiny += "00000000000000000001"
    assert_equal(_float_of(tiny), -1e-77)

    assert_equal(
        _float_of("2.225073858507201136057409796709131975934819546351645648"),
        2.225073858507201,
    )
    assert_equal(_float_of("9007199254740993.0"), 9007199254740992.0)
    print("  test_long_mantissa_is_correctly_rounded passed")


def test_overflow_is_an_error() raises:
    """`1e400` is not `inf`: `inf` cannot be serialized back to JSON."""
    with assert_raises(contains="out of range"):
        _ = scan_number("1e400".as_bytes(), 0, 5)
    with assert_raises(contains="out of range"):
        _ = scan_number("-1e400".as_bytes(), 0, 6)
    with assert_raises(contains="out of range"):
        _ = scan_number("1e999999".as_bytes(), 0, 8)
    print("  test_overflow_is_an_error passed")


def test_matches_reference_over_the_double_range() raises:
    """Differential check against a correctly rounded parser.

    The interesting half of this is the exact fast path: whenever the
    gate says a significand and exponent are small enough for exact
    arithmetic, the answer has to match the reference bit for bit, or
    the gate is letting through inputs it should not.
    """
    var mismatches = 0
    for _ in range(200000):
        var bits = random_ui64(0, UInt64.MAX)
        if (bits >> 52) & 0x7FF == 0x7FF:
            continue
        var text = String(bitcast[DType.float64](bits))
        var scanned = parse_float_span(text.as_bytes(), 0, text.byte_length())
        if scanned[0] != _reference(text):
            mismatches += 1
            if mismatches < 4:
                print("    mismatch:", text, "->", scanned[0])
    assert_equal(mismatches, 0)
    print("  test_matches_reference_over_the_double_range passed")


def test_matches_reference_on_ordinary_decimals() raises:
    """The same, for values written the way documents actually hold them.

    These land on the exact path almost every time, which is the point:
    it is the path that has to be right for real data.
    """
    var mismatches = 0
    for _ in range(100000):
        var text = String(random_float64() * 1000.0)
        var scanned = parse_float_span(text.as_bytes(), 0, text.byte_length())
        if scanned[0] != _reference(text):
            mismatches += 1
    assert_equal(mismatches, 0)
    print("  test_matches_reference_on_ordinary_decimals passed")


def test_written_decimals_read_back() raises:
    """Hand-written decimals parse to the value they denote.

    Spot checks with values verified against an independent
    correctly-rounded parser, so this does not lean on any Mojo
    formatter being right.
    """
    assert_equal(_float_of("2.9245645358754477e+17"), 2.9245645358754477e17)
    assert_equal(_float_of("-2.6752382909860068e+16"), -2.6752382909860068e16)
    assert_equal(_float_of("9.884471094837134e+17"), 9.884471094837134e17)
    assert_equal(_float_of("123456789.123456789"), 123456789.12345679)
    assert_equal(_float_of("0.30000000000000004"), 0.30000000000000004)
    print("  test_written_decimals_read_back passed")


# ===================================================================
# Typed integer reads
# ===================================================================


def test_parse_int_checked_ranges() raises:
    var a = parse_int_checked[DType.int32]("2147483647".as_bytes(), 0, 10)
    assert_equal(a[0], Int32.MAX)

    var b = parse_int_checked[DType.uint8]("255".as_bytes(), 0, 3)
    assert_equal(b[0], UInt8(255))

    with assert_raises(contains="out of range"):
        _ = parse_int_checked[DType.int32]("2147483648".as_bytes(), 0, 10)
    with assert_raises(contains="out of range"):
        _ = parse_int_checked[DType.int8]("-129".as_bytes(), 0, 4)
    with assert_raises(contains="out of range"):
        _ = parse_int_checked[DType.uint8]("256".as_bytes(), 0, 3)
    with assert_raises(contains="unsigned"):
        _ = parse_int_checked[DType.uint32]("-1".as_bytes(), 0, 2)
    print("  test_parse_int_checked_ranges passed")


def test_parse_int_checked_rejects_floats() raises:
    """`1.5` into an integer field is an error, not a truncation.

    Dropping the fraction loses data the caller asked to keep, and
    does it silently.
    """
    with assert_raises(contains="expected an integer"):
        _ = parse_int_checked[DType.int64]("1.5".as_bytes(), 0, 3)
    with assert_raises(contains="expected an integer"):
        _ = parse_int_checked[DType.int64]("1e3".as_bytes(), 0, 3)
    with assert_raises(contains="invalid number"):
        _ = parse_int_checked[DType.int64]("-".as_bytes(), 0, 1)
    print("  test_parse_int_checked_rejects_floats passed")


def main() raises:
    print("Grammar:")
    test_rejects_malformed_numbers()
    test_rejects_leading_zeros()
    test_stops_without_consuming()
    test_accepts_exponent_spellings()
    print()

    print("Integer range:")
    test_int64_boundaries()
    test_above_int64_is_unsigned()
    test_beyond_uint64_becomes_float()
    print()

    print("Float conversion:")
    test_float_edges()
    test_long_mantissa_is_correctly_rounded()
    test_overflow_is_an_error()
    test_matches_reference_over_the_double_range()
    test_matches_reference_on_ordinary_decimals()
    test_written_decimals_read_back()
    print()

    print("Typed integer reads:")
    test_parse_int_checked_ranges()
    test_parse_int_checked_rejects_floats()
    print()

    print("All number parsing tests passed!")
