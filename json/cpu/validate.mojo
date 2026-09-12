# Byte-level validators shared by both readers.
#
# Stage 2 walks a structural index; the typed reader walks bytes. They
# have to agree about what a valid string is, which they will not do
# for long if each carries its own copy of the rules. Everything here
# is a pure function over a span, with no knowledge of tapes, values or
# documents, so both callers can use it and the tests can drive it
# directly.
#
# Scanning first, deciding second
# -------------------------------
# `scan_string_body` makes one SIMD pass and returns flags: does this
# string contain an escape, a character below U+0020, a byte above
# ASCII. On the overwhelmingly common case -- plain ASCII with nothing
# to escape -- that is the whole cost, and the caller skips the escape
# validator, the UTF-8 validator and the unescaper in one branch.
# Everything after that pass only runs for strings that need it.
#
# UTF-8 is validated per string rather than over the whole document,
# because outside a string the grammar already rejects every byte at
# or above 0x80 as a stray character. Validating a document twice to
# reach the same verdict is a pass nobody needs.

from std.bit import count_trailing_zeros
from std.collections.string._utf8 import _is_valid_utf8
from std.memory.unsafe import pack_bits
from std.sys import simd_byte_width

from ..unicode import hex_digit_value, is_high_surrogate, is_low_surrogate

comptime _BLOCK: Int = simd_byte_width()
"""One native SIMD register, so a chunk is one instruction everywhere."""

comptime _QUOTE = UInt8(ord('"'))
comptime _BACKSLASH = UInt8(ord("\\"))
comptime _SPACE = UInt8(ord(" "))


# ---------------------------------------------------------------------------
# Whitespace
# ---------------------------------------------------------------------------


@always_inline
def is_ws(b: UInt8) -> Bool:
    """The four bytes RFC 8259 section 2 allows between tokens."""
    return (
        b == UInt8(ord(" "))
        or b == UInt8(ord("\t"))
        or b == UInt8(ord("\n"))
        or b == UInt8(ord("\r"))
    )


@always_inline
def skip_ws(bytes: Span[UInt8, _], start: Int, end: Int) -> Int:
    """Offset of the first non-whitespace byte at or after `start`.

    A scalar prelude of four bytes handles compact documents, where
    there is rarely more than one whitespace byte between tokens and
    the SIMD setup would not pay for itself. Only a longer run falls
    into the vector loop, which finds the first non-whitespace byte
    with `pack_bits` and a trailing-zero count instead of stepping.
    """
    var i = start
    var ptr = bytes.unsafe_ptr()

    var scalar_stop = min(end, start + 4)
    while i < scalar_stop and is_ws(ptr[unsafe_offset=i]):
        i += 1
    if i < scalar_stop:
        return i

    var stop = end - _BLOCK
    while i <= stop:
        var chunk = ptr.unsafe_load[width=_BLOCK](i)
        var ws_mask = (
            chunk.eq(UInt8(ord(" ")))
            | chunk.eq(UInt8(ord("\t")))
            | chunk.eq(UInt8(ord("\n")))
            | chunk.eq(UInt8(ord("\r")))
        )
        comptime if _BLOCK == 16:
            var bits = pack_bits[dtype=DType.uint16](ws_mask)
            if bits != UInt16(0xFFFF):
                return i + Int(count_trailing_zeros(~bits))
        elif _BLOCK == 32:
            var bits = pack_bits[dtype=DType.uint32](ws_mask)
            if bits != UInt32(0xFFFF_FFFF):
                return i + Int(count_trailing_zeros(~bits))
        elif _BLOCK == 64:
            var bits = pack_bits[dtype=DType.uint64](ws_mask)
            if bits != UInt64(0xFFFF_FFFF_FFFF_FFFF):
                return i + Int(count_trailing_zeros(~bits))
        else:
            comptime assert False, "unsupported simd_byte_width()"
        i += _BLOCK

    while i < end and is_ws(ptr[unsafe_offset=i]):
        i += 1
    return i


# ---------------------------------------------------------------------------
# String body classification
# ---------------------------------------------------------------------------

comptime STR_PLAIN: UInt8 = 0
"""ASCII, no escapes, nothing below U+0020: copy it and move on."""
comptime STR_ESCAPE: UInt8 = 1
"""Contains a backslash, so the escapes need validating and expanding."""
comptime STR_CONTROL: UInt8 = 2
"""Contains a byte below U+0020, which section 7 forbids unescaped."""
comptime STR_NON_ASCII: UInt8 = 4
"""Contains a byte at or above 0x80, so the UTF-8 needs validating."""


@always_inline
def scan_string_body(bytes: Span[UInt8, _], start: Int, end: Int) -> UInt8:
    """Classify a string's bytes in one pass.

    Returns the OR of the `STR_*` flags. `STR_PLAIN` (zero) is the
    answer for most strings in most documents, and it is the answer
    that lets the caller take a zero-copy slice.
    """
    var flags: UInt8 = 0
    var n = end - start
    if n <= 0:
        return flags
    var ptr = bytes.unsafe_ptr()
    var i = start
    var stop = end - _BLOCK
    while i <= stop:
        var chunk = ptr.unsafe_load[width=_BLOCK](i)
        if chunk.eq(_BACKSLASH).reduce_or():
            flags |= STR_ESCAPE
        if chunk.lt(_SPACE).reduce_or():
            flags |= STR_CONTROL
        if chunk.ge(UInt8(0x80)).reduce_or():
            flags |= STR_NON_ASCII
        i += _BLOCK
    while i < end:
        var c = ptr[unsafe_offset=i]
        if c == _BACKSLASH:
            flags |= STR_ESCAPE
        elif c < _SPACE:
            flags |= STR_CONTROL
        elif c >= UInt8(0x80):
            flags |= STR_NON_ASCII
        i += 1
    return flags


def find_control_char(bytes: Span[UInt8, _], start: Int, end: Int) -> Int:
    """Offset of the first byte below U+0020, or -1.

    Only called once a scan has already said there is one, so it can
    afford to be a plain loop: its job is to name the position for the
    error message.
    """
    for i in range(start, end):
        if bytes[i] < _SPACE:
            return i
    return -1


# ---------------------------------------------------------------------------
# UTF-8
# ---------------------------------------------------------------------------


def is_valid_utf8(bytes: Span[UInt8, _]) -> Bool:
    """Whether `bytes` is well-formed UTF-8 (RFC 3629).

    Rejects overlong encodings, encoded surrogates, anything above
    U+10FFFF, and truncated or orphaned continuation bytes. Wraps the
    stdlib's vectorized validator so there is one place to change if
    that moves.
    """
    return _is_valid_utf8(bytes)


def first_invalid_utf8(bytes: Span[UInt8, _], start: Int, end: Int) -> Int:
    """Offset of the first byte that cannot begin a valid sequence.

    Walks the sequence structure scalar, which is fine because this
    only runs on the error path, after the vectorized check has
    already said the span is bad. Its job is to point at the byte.
    """
    var i = start
    while i < end:
        var c = Int(bytes[i])
        var length: Int
        if c < 0x80:
            i += 1
            continue
        elif c >= 0xC2 and c <= 0xDF:
            length = 2
        elif c >= 0xE0 and c <= 0xEF:
            length = 3
        elif c >= 0xF0 and c <= 0xF4:
            length = 4
        else:
            return i
        if i + length > end:
            return i
        if not is_valid_utf8(bytes[i : i + length]):
            return i
        i += length
    return -1


# ---------------------------------------------------------------------------
# Escapes
# ---------------------------------------------------------------------------

comptime ESC_OK: UInt8 = 0
comptime ESC_TRAILING_BACKSLASH: UInt8 = 1
"""A backslash with nothing after it."""
comptime ESC_BAD_CHAR: UInt8 = 2
"""A backslash followed by something other than `"\\/bfnrtu`."""
comptime ESC_BAD_HEX: UInt8 = 3
"""`\\u` without exactly four hexadecimal digits after it."""
comptime ESC_LONE_SURROGATE: UInt8 = 4
"""An unpaired surrogate escape; only reported when `ijson` is set."""


def validate_escapes(
    bytes: Span[UInt8, _],
    start: Int,
    end: Int,
    ijson: Bool,
    mut err_pos: Int,
) -> UInt8:
    """Check every escape in a string body.

    The four hexadecimal digits after `\\u` were never checked before,
    and an escape that failed to parse was kept as literal text rather
    than reported, so `"\\uqqqq"` and `"\\u00A"` both parsed
    successfully into nonsense.

    A surrogate escape is paired here rather than in the unescaper, so
    that the decision about an unpaired one is made once. RFC 8259
    section 8.2 permits it -- the unescaper substitutes U+FFFD -- and
    RFC 7493 section 2.1 forbids it, which is what `ijson` selects.

    Args:
        bytes: The document.
        start: First byte of the string body.
        end: One past the last byte of the string body.
        ijson: Whether to reject unpaired surrogate escapes.
        err_pos: Set to the offset of the offending backslash on failure.

    Returns:
        `ESC_OK`, or one of the `ESC_*` reasons.
    """
    var j = start
    while j < end:
        if bytes[j] != _BACKSLASH:
            j += 1
            continue
        err_pos = j
        if j + 1 >= end:
            return ESC_TRAILING_BACKSLASH
        var esc = bytes[j + 1]
        if esc == UInt8(ord("u")):
            if j + 6 > end:
                return ESC_BAD_HEX
            var code = 0
            for k in range(j + 2, j + 6):
                var digit = hex_digit_value(bytes[k])
                if digit < 0:
                    return ESC_BAD_HEX
                code = code * 16 + digit
            j += 6
            if is_low_surrogate(code):
                if ijson:
                    return ESC_LONE_SURROGATE
                continue
            if not is_high_surrogate(code):
                continue
            # A high surrogate must be followed by `\uDC00`-`\uDFFF`.
            if (
                j + 6 > end
                or bytes[j] != _BACKSLASH
                or bytes[j + 1] != UInt8(ord("u"))
            ):
                if ijson:
                    return ESC_LONE_SURROGATE
                continue
            var low = 0
            for k in range(j + 2, j + 6):
                var digit = hex_digit_value(bytes[k])
                if digit < 0:
                    return ESC_BAD_HEX
                low = low * 16 + digit
            if not is_low_surrogate(low):
                if ijson:
                    return ESC_LONE_SURROGATE
                continue
            j += 6
            continue
        if (
            esc != _QUOTE
            and esc != _BACKSLASH
            and esc != UInt8(ord("/"))
            and esc != UInt8(ord("b"))
            and esc != UInt8(ord("f"))
            and esc != UInt8(ord("n"))
            and esc != UInt8(ord("r"))
            and esc != UInt8(ord("t"))
        ):
            return ESC_BAD_CHAR
        j += 2
    return ESC_OK
