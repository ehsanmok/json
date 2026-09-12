# Number scanning and conversion for the JSON grammar.
#
# One tokenizer serves both readers. Stage 2 calls it while walking the
# structural index; the typed reader calls it while walking bytes. They
# used to disagree about what a number is, which is how a bare `-`
# parsed as 0 on one path and raised on the other.
#
# Grammar (RFC 8259 section 6), enforced exactly:
#
#   number = [ "-" ] int [ frac ] [ exp ]
#   int    = "0" / ( digit1-9 *DIGIT )
#   frac   = "." 1*DIGIT
#   exp    = ("e" / "E") [ "-" / "+" ] 1*DIGIT
#
# So `-`, `01`, `1.`, `.5`, `+1`, `1e`, `1e+`, `0.e1`, `1eE2` and
# `0e+-1` are all rejected, and the scanner stops at the first byte
# that cannot continue the production without consuming it -- `1+2`
# yields the number `1` and leaves `+` for the caller, which is what
# lets the caller report "expected ',' or ']'" rather than a number
# error.
#
# Conversion
# ----------
# Digits are accumulated as they are validated, in one pass. A 20-digit
# run is re-accumulated afterwards with a checked multiply, because 20
# digits may or may not fit `UInt64`. Anything longer is a float.
#
# Floats take Clinger's fast path when the significand and the
# exponent are both small enough that `Float64` arithmetic is exact --
# a significand under 2**53 and |exponent| within the exactly
# representable powers of ten. That covers the shapes documents
# actually carry: prices, coordinates, durations, anything with at
# most fifteen significant digits.
#
# Everything else goes to `strtod`, which is correctly rounded.
# Eisel-Lemire would be the faster answer and the stdlib exposes an
# implementation, but it disagrees with a correctly rounded parser by
# one unit in the last place on about one random double in a thousand
# -- `9.884471094837134e+17` reads back as `...133e+17` -- so a value
# written by this library would not always read back as itself. The
# differential test in `tests/test_number_parse.mojo` is what caught
# that and is what will catch it if the fast path is ever widened
# incorrectly.
#
# `strtod` needs a NUL terminator, which a span does not have, so the
# token is copied into a stack buffer first. Only a token longer than
# that buffer allocates, and at that length the number is already
# carrying more digits than a `Float64` can hold.
#
# A value that overflows to infinity is an error rather than `inf`:
# `inf` is not JSON, so accepting it produces a document that cannot be
# serialized back.

from std.collections.string._parsing_numbers.parsing_floats import (
    can_use_clinger_fast_path,
    clinger_fast_path,
)
from std.ffi import external_call
from std.memory import unsafe_memcpy
from std.utils.numerics import isinf

comptime _STRTOD_BUFFER = 64
"""Stack bytes for the `strtod` copy; longer tokens allocate."""

# --- token kinds -----------------------------------------------------------

comptime NUM_INT: UInt8 = 0
"""Integer grammar, value fits `Int64`."""
comptime NUM_UINT: UInt8 = 1
"""Integer grammar, value exceeds `Int64.MAX` but fits `UInt64`."""
comptime NUM_FLOAT: UInt8 = 2
"""Has a fraction or exponent, or an integer too large for `UInt64`."""
comptime NUM_INVALID: UInt8 = 3
"""Not a number; `end` is the offset of the offending byte."""

# --- rejection reasons -----------------------------------------------------

comptime NUM_ERR_NONE: UInt8 = 0
comptime NUM_ERR_GRAMMAR: UInt8 = 1
"""A byte appeared where the grammar does not allow it."""
comptime NUM_ERR_LEADING_ZERO: UInt8 = 2
"""`01`, `-00`: a zero followed by another digit."""
comptime NUM_ERR_RANGE: UInt8 = 3
"""Magnitude overflows `Float64`."""

comptime _MAX_EXP = 1000000
"""Exponents saturate here; past it the result is zero or an error anyway."""


@fieldwise_init
struct NumberToken(Copyable, Movable):
    """One scanned number: its kind, its value, and where it ended."""

    var kind: UInt8
    var err: UInt8
    var end: Int
    var int_value: Int64
    var uint_value: UInt64
    var float_value: Float64

    @staticmethod
    def invalid(reason: UInt8, at: Int) -> NumberToken:
        return NumberToken(NUM_INVALID, reason, at, 0, 0, 0.0)


@always_inline
def _is_digit(c: UInt8) -> Bool:
    return c >= UInt8(ord("0")) and c <= UInt8(ord("9"))


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _strtod(bytes: Span[UInt8, _], start: Int, end: Int) -> Float64:
    """Correctly rounded conversion of an already-validated token.

    The token is NUL-terminated in a stack buffer so the common case
    does not allocate. `strtod` reads the decimal point from the
    locale; nothing in this library calls `setlocale`, so the C locale
    applies and the separator is `.` as JSON requires.
    """
    var n = end - start
    if n < _STRTOD_BUFFER:
        var buffer = InlineArray[UInt8, _STRTOD_BUFFER](uninitialized=True)
        unsafe_memcpy(
            dest=buffer.unsafe_ptr(),
            src=bytes.unsafe_ptr().unsafe_offset(start),
            count=n,
        )
        buffer[n] = 0
        return external_call["strtod", Float64](buffer.unsafe_ptr(), Int(0))
    var text = String(unsafe_from_utf8=bytes[start:end])
    var c_str = text.as_c_string_slice()
    return external_call["strtod", Float64](c_str.unsafe_ptr(), Int(0))


def _to_float(
    negative: Bool,
    significand: UInt64,
    exponent: Int64,
    truncated: Bool,
    bytes: Span[UInt8, _],
    start: Int,
    end: Int,
) raises -> Float64:
    """Combine a scanned (significand, power of ten) into a `Float64`."""
    var value: Float64
    if not truncated and can_use_clinger_fast_path(significand, exponent):
        value = clinger_fast_path(significand, exponent)
        if negative:
            value = -value
    else:
        # `strtod` reads the sign off the token itself.
        value = _strtod(bytes, start, end)
    if isinf(value):
        raise Error("number out of range")
    return value


def _checked_uint(
    bytes: Span[UInt8, _], start: Int, end: Int
) -> Tuple[UInt64, Bool]:
    """Re-accumulate a digit run, reporting whether it fits `UInt64`.

    Only the 20-digit case needs this: 19 digits always fit, 21 never
    do, and 20 sometimes do.
    """
    var acc: UInt64 = 0
    for i in range(start, end):
        var d = UInt64(bytes[i]) - UInt64(ord("0"))
        if acc > (UInt64.MAX - d) // 10:
            return (UInt64(0), False)
        acc = acc * 10 + d
    return (acc, True)


# ---------------------------------------------------------------------------
# The scanner
# ---------------------------------------------------------------------------


def scan_number(
    bytes: Span[UInt8, _], start: Int, limit: Int
) raises -> NumberToken:
    """Scan one JSON number from `start`, stopping at the first byte
    that cannot extend it.

    Args:
        bytes: The document.
        start: Offset of the first byte of the number.
        limit: One past the last byte the scanner may read.

    Returns:
        A token whose `kind` says how to read the value, or `NUM_INVALID`
        with `end` pointing at the byte that broke the grammar.

    Raises:
        If the magnitude overflows `Float64`.
    """
    var pos = start
    var negative = False
    if pos < limit and bytes[pos] == UInt8(ord("-")):
        negative = True
        pos += 1
    if pos >= limit:
        return NumberToken.invalid(NUM_ERR_GRAMMAR, pos)

    # Digits are accumulated as they are validated. Walking them twice
    # -- once for the grammar, once for the value -- was most of what
    # scanning a short number cost, and the second walk re-read bytes
    # the first had just touched.
    #
    # Leading zeros do not count toward significance, which is why the
    # counter only advances once the accumulator is non-zero.
    var significand: UInt64 = 0
    var digits = 0
    var truncated = False
    var exponent: Int64 = 0

    # --- int ---------------------------------------------------------------
    var int_start = pos
    var c = bytes[pos]
    if c == UInt8(ord("0")):
        pos += 1
        if pos < limit and _is_digit(bytes[pos]):
            return NumberToken.invalid(NUM_ERR_LEADING_ZERO, pos)
    elif _is_digit(c):
        while pos < limit:
            var d = bytes[pos]
            if not _is_digit(d):
                break
            if digits < 19:
                significand = significand * 10 + UInt64(d) - UInt64(ord("0"))
                if significand != 0:
                    digits += 1
            else:
                truncated = True
                exponent += 1
            pos += 1
    else:
        return NumberToken.invalid(NUM_ERR_GRAMMAR, pos)
    var int_end = pos

    # --- frac --------------------------------------------------------------
    var is_float = False
    if pos < limit and bytes[pos] == UInt8(ord(".")):
        is_float = True
        pos += 1
        if pos >= limit or not _is_digit(bytes[pos]):
            return NumberToken.invalid(NUM_ERR_GRAMMAR, pos)
        while pos < limit:
            var d = bytes[pos]
            if not _is_digit(d):
                break
            if digits < 19:
                significand = significand * 10 + UInt64(d) - UInt64(ord("0"))
                if significand != 0:
                    digits += 1
                exponent -= 1
            else:
                truncated = True
            pos += 1

    # --- exp ---------------------------------------------------------------
    if pos < limit and (
        bytes[pos] == UInt8(ord("e")) or bytes[pos] == UInt8(ord("E"))
    ):
        is_float = True
        pos += 1
        var exp_negative = False
        if pos < limit and (
            bytes[pos] == UInt8(ord("+")) or bytes[pos] == UInt8(ord("-"))
        ):
            exp_negative = bytes[pos] == UInt8(ord("-"))
            pos += 1
        if pos >= limit or not _is_digit(bytes[pos]):
            return NumberToken.invalid(NUM_ERR_GRAMMAR, pos)
        var acc: Int64 = 0
        while pos < limit and _is_digit(bytes[pos]):
            if acc < _MAX_EXP:
                acc = acc * 10 + Int64(Int(bytes[pos]) - ord("0"))
            pos += 1
        exponent += -acc if exp_negative else acc

    if is_float:
        var f = _to_float(
            negative, significand, exponent, truncated, bytes, start, pos
        )
        return NumberToken(NUM_FLOAT, NUM_ERR_NONE, pos, 0, 0, f)

    # --- integer -----------------------------------------------------------
    var run = int_end - int_start
    if run > 19:
        if run == 20:
            var checked = _checked_uint(bytes, int_start, int_end)
            if checked[1]:
                significand = checked[0]
                truncated = False
            else:
                truncated = True
        else:
            truncated = True

    if truncated:
        # Beyond `UInt64`. RFC 8259 leaves the range to the
        # implementation; the widely compatible answer is the nearest
        # `Float64`, which is what a JavaScript reader would produce.
        var f = _to_float(
            negative, significand, exponent, True, bytes, start, pos
        )
        return NumberToken(NUM_FLOAT, NUM_ERR_NONE, pos, 0, 0, f)

    if negative:
        if significand > UInt64(1) << 63:
            var f = _to_float(
                negative, significand, exponent, False, bytes, start, pos
            )
            return NumberToken(NUM_FLOAT, NUM_ERR_NONE, pos, 0, 0, f)
        if significand == UInt64(1) << 63:
            return NumberToken(NUM_INT, NUM_ERR_NONE, pos, Int64.MIN, 0, 0.0)
        return NumberToken(
            NUM_INT, NUM_ERR_NONE, pos, -Int64(significand), 0, 0.0
        )

    if significand > UInt64(Int64.MAX):
        return NumberToken(NUM_UINT, NUM_ERR_NONE, pos, 0, significand, 0.0)
    return NumberToken(NUM_INT, NUM_ERR_NONE, pos, Int64(significand), 0, 0.0)


# ---------------------------------------------------------------------------
# Typed wrappers
# ---------------------------------------------------------------------------


def parse_float_span(
    bytes: Span[UInt8, _], start: Int, limit: Int
) raises -> Tuple[Float64, Int]:
    """Read one number as a `Float64`. Returns (value, end offset)."""
    var token = scan_number(bytes, start, limit)
    if token.kind == NUM_INVALID:
        raise Error("invalid number")
    if token.kind == NUM_FLOAT:
        return (token.float_value, token.end)
    if token.kind == NUM_UINT:
        return (Float64(token.uint_value), token.end)
    return (Float64(token.int_value), token.end)


def parse_int_checked[
    dtype: DType
](bytes: Span[UInt8, _], start: Int, limit: Int) raises -> Tuple[
    Scalar[dtype], Int
]:
    """Read one number as an integer of `dtype`, or raise.

    A fractional or exponent part is a type error, not a truncation:
    silently turning `1.5` into `1` loses data the caller asked to
    keep. A magnitude outside `dtype` is an error for the same reason.

    Parameters:
        dtype: The integer type to produce.

    Args:
        bytes: The document.
        start: Offset of the first byte of the number.
        limit: One past the last byte the scanner may read.

    Returns:
        The value and the offset one past the number.

    Raises:
        If the token is not an integer or does not fit `dtype`.
    """
    comptime assert (
        dtype.is_integral()
    ), "parse_int_checked needs an integer type"
    var token = scan_number(bytes, start, limit)
    if token.kind == NUM_INVALID:
        raise Error("invalid number")
    if token.kind == NUM_FLOAT:
        raise Error(
            "expected an integer, got a number with a fraction or exponent"
        )

    comptime if dtype.is_signed():
        var v: Int64
        if token.kind == NUM_UINT:
            raise Error("integer out of range for " + String(dtype))
        v = token.int_value
        if v < Int64(Scalar[dtype].MIN_FINITE) or v > Int64(
            Scalar[dtype].MAX_FINITE
        ):
            raise Error("integer out of range for " + String(dtype))
        return (Scalar[dtype](v), token.end)
    else:
        var u: UInt64
        if token.kind == NUM_UINT:
            u = token.uint_value
        else:
            if token.int_value < 0:
                raise Error("negative integer in an unsigned field")
            u = UInt64(token.int_value)
        if u > UInt64(Scalar[dtype].MAX_FINITE):
            raise Error("integer out of range for " + String(dtype))
        return (Scalar[dtype](u), token.end)
