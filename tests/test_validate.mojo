"""Tests for the shared byte validators.

These are the rules both readers have to agree on, so they are tested
against the byte sequences rather than through either parser. The
UTF-8 cases are the classes RFC 3629 calls out -- overlong forms,
encoded surrogates, values past U+10FFFF, truncated and orphaned
continuation bytes -- because those are what an attacker sends and
what a naive length-driven decoder lets through.
"""

from std.collections import List
from std.random import random_ui64
from std.testing import assert_equal, assert_false, assert_true

from json.cpu.validate import (
    ESC_BAD_CHAR,
    ESC_BAD_HEX,
    ESC_LONE_SURROGATE,
    ESC_OK,
    ESC_TRAILING_BACKSLASH,
    STR_CONTROL,
    STR_ESCAPE,
    STR_NON_ASCII,
    STR_PLAIN,
    find_control_char,
    first_invalid_utf8,
    is_valid_utf8,
    is_ws,
    scan_string_body,
    skip_ws,
    validate_escapes,
)


def _bytes(values: List[Int]) -> List[UInt8]:
    var out = List[UInt8](capacity=len(values))
    for i in range(len(values)):
        out.append(UInt8(values[i]))
    return out^


def _flags(text: String) -> UInt8:
    return scan_string_body(text.as_bytes(), 0, text.byte_length())


def _escapes(text: String, ijson: Bool = False) -> UInt8:
    var pos = 0
    return _escapes_at(text, ijson, pos)


def _escapes_at(text: String, ijson: Bool, mut pos: Int) -> UInt8:
    return validate_escapes(text.as_bytes(), 0, text.byte_length(), ijson, pos)


# ===================================================================
# Whitespace
# ===================================================================


def test_whitespace_set() raises:
    """Exactly four bytes, not the wider set `isspace` would accept."""
    assert_true(is_ws(UInt8(ord(" "))))
    assert_true(is_ws(UInt8(ord("\t"))))
    assert_true(is_ws(UInt8(ord("\n"))))
    assert_true(is_ws(UInt8(ord("\r"))))
    assert_false(is_ws(UInt8(0x0B)))  # vertical tab
    assert_false(is_ws(UInt8(0x0C)))  # form feed
    assert_false(is_ws(UInt8(0x00)))
    print("  test_whitespace_set passed")


def test_skip_ws_scalar_and_vector() raises:
    """The scalar prelude and the vector body must agree.

    A run long enough to reach the vector loop is the case the prelude
    was added to skip, so both have to be exercised or one of them
    rots.
    """
    var compact = String('{"a":1}')
    assert_equal(skip_ws(compact.as_bytes(), 0, compact.byte_length()), 0)

    var one = String(' {"a":1}')
    assert_equal(skip_ws(one.as_bytes(), 0, one.byte_length()), 1)

    var long_run = String(" " * 200 + "x")
    assert_equal(skip_ws(long_run.as_bytes(), 0, long_run.byte_length()), 200)

    var mixed = String(" \t\r\n" * 40 + "y")
    assert_equal(skip_ws(mixed.as_bytes(), 0, mixed.byte_length()), 160)

    var all_ws = String("   ")
    assert_equal(skip_ws(all_ws.as_bytes(), 0, 3), 3)
    print("  test_skip_ws_scalar_and_vector passed")


# ===================================================================
# String classification
# ===================================================================


def test_scan_string_body_flags() raises:
    assert_equal(_flags("plain ascii"), STR_PLAIN)
    assert_equal(_flags(""), STR_PLAIN)
    assert_equal(_flags("a\\nb"), STR_ESCAPE)
    assert_equal(_flags("café"), STR_NON_ASCII)
    assert_equal(_flags("a\tb"), STR_CONTROL)
    assert_equal(_flags("café \\u0041"), STR_ESCAPE | STR_NON_ASCII)
    print("  test_scan_string_body_flags passed")


def test_scan_string_body_past_one_vector() raises:
    """Flags must survive a body longer than one SIMD register."""
    var long_plain = String("x" * 300)
    assert_equal(scan_string_body(long_plain.as_bytes(), 0, 300), STR_PLAIN)

    var late_escape = String("x" * 300 + "\\n")
    assert_equal(scan_string_body(late_escape.as_bytes(), 0, 302), STR_ESCAPE)

    var late_control = String("x" * 300 + "\t")
    assert_equal(scan_string_body(late_control.as_bytes(), 0, 301), STR_CONTROL)
    print("  test_scan_string_body_past_one_vector passed")


def test_find_control_char() raises:
    var text = String("abc\tdef")
    assert_equal(find_control_char(text.as_bytes(), 0, 7), 3)
    var clean = String("abcdef")
    assert_equal(find_control_char(clean.as_bytes(), 0, 6), -1)
    print("  test_find_control_char passed")


# ===================================================================
# UTF-8
# ===================================================================


def test_utf8_accepts_well_formed() raises:
    assert_true(is_valid_utf8(String("ascii").as_bytes()))
    assert_true(is_valid_utf8(String("café").as_bytes()))
    assert_true(is_valid_utf8(String("€").as_bytes()))
    assert_true(is_valid_utf8(String("\U0001F600").as_bytes()))
    assert_true(is_valid_utf8(String("").as_bytes()))
    print("  test_utf8_accepts_well_formed passed")


def test_utf8_rejects_malformed() raises:
    """One case per class RFC 3629 section 3 rules out."""
    var cases: List[List[Int]] = [
        [0xC0, 0x80],  # overlong NUL
        [0xC1, 0xBF],  # overlong
        [0xE0, 0x80, 0x80],  # overlong 3-byte
        [0xF0, 0x80, 0x80, 0x80],  # overlong 4-byte
        [0xED, 0xA0, 0x80],  # U+D800 encoded as UTF-8
        [0xED, 0xBF, 0xBF],  # U+DFFF encoded as UTF-8
        [0xF4, 0x90, 0x80, 0x80],  # above U+10FFFF
        [0xF5, 0x80, 0x80, 0x80],  # above U+10FFFF
        [0xE2, 0x82],  # truncated 3-byte
        [0xF0, 0x9F, 0x98],  # truncated 4-byte
        [0x80],  # orphaned continuation byte
        [0xBF],  # orphaned continuation byte
        [0xFE],  # never valid
        [0xFF],  # never valid
        [0xC3],  # lead byte with nothing after it
    ]
    for i in range(len(cases)):
        var raw = _bytes(cases[i])
        assert_false(
            is_valid_utf8(Span(raw)),
            "should be invalid UTF-8: case " + String(i),
        )
    print("  test_utf8_rejects_malformed passed")


def test_first_invalid_utf8_points_at_the_byte() raises:
    var raw = _bytes([0x61, 0x62, 0xC0, 0x80, 0x63])
    assert_equal(first_invalid_utf8(Span(raw), 0, len(raw)), 2)

    var good = _bytes([0x61, 0xC3, 0xA9, 0x62])
    assert_equal(first_invalid_utf8(Span(good), 0, len(good)), -1)
    print("  test_first_invalid_utf8_points_at_the_byte passed")


def test_utf8_agrees_with_the_stdlib_constructor() raises:
    """Differential check over random bytes.

    `StringSlice(from_utf8=...)` validates and raises, so it is an
    independent oracle for the same question.
    """
    var disagreements = 0
    for _ in range(20000):
        var n = Int(random_ui64(1, 6))
        var raw = List[UInt8](capacity=n)
        for _ in range(n):
            raw.append(UInt8(random_ui64(0, 255)))
        var ours = is_valid_utf8(Span(raw))
        var theirs = True
        try:
            _ = StringSlice(from_utf8=Span(raw))
        except:
            theirs = False
        if ours != theirs:
            disagreements += 1
    assert_equal(disagreements, 0)
    print("  test_utf8_agrees_with_the_stdlib_constructor passed")


# ===================================================================
# Escapes
# ===================================================================


def test_escapes_accepts_the_eight_short_forms() raises:
    assert_equal(_escapes('\\" \\\\ \\/ \\b \\f \\n \\r \\t'), ESC_OK)
    assert_equal(_escapes("no escapes here"), ESC_OK)
    print("  test_escapes_accepts_the_eight_short_forms passed")


def test_escapes_rejects_unknown_letters() raises:
    """`\\x` and friends are not escapes, however common they look."""
    for text in ["\\x41", "\\a", "\\U0041", "\\'", "\\0"]:
        var pos = 0
        assert_equal(_escapes_at(text, False, pos), ESC_BAD_CHAR, text)
    print("  test_escapes_rejects_unknown_letters passed")


def test_escapes_rejects_trailing_backslash() raises:
    assert_equal(_escapes("abc\\"), ESC_TRAILING_BACKSLASH)
    print("  test_escapes_rejects_trailing_backslash passed")


def test_escapes_validates_hex_digits() raises:
    """The four digits after `\\u` were never checked.

    A malformed escape used to survive as literal text, so `"\\uqqqq"`
    parsed successfully into the characters `\\uqqqq`.
    """
    for text in ["\\uqqqq", "\\u00A", "\\u", "\\u12", "\\uD834\\uDd"]:
        var pos = 0
        assert_equal(_escapes_at(text, False, pos), ESC_BAD_HEX, text)
    assert_equal(_escapes("\\u0041"), ESC_OK)
    assert_equal(_escapes("\\uFFFF"), ESC_OK)
    assert_equal(_escapes("\\uffff"), ESC_OK)
    print("  test_escapes_validates_hex_digits passed")


def test_escapes_pairs_surrogates() raises:
    """A pair is one character; an unpaired one is a question of mode.

    RFC 8259 section 8.2 allows an unpaired surrogate and leaves the
    result to the reader; RFC 7493 section 2.1 forbids it. The
    decision lives here so both readers make it the same way.
    """
    assert_equal(_escapes("\\uD834\\uDD1E"), ESC_OK)
    assert_equal(_escapes("\\uD834\\uDD1E", ijson=True), ESC_OK)

    # Unpaired: allowed by default, rejected under I-JSON.
    for text in ["\\uD800", "\\uDC00", "\\uD800\\u0041", "\\uD800abc"]:
        var lenient = 0
        assert_equal(_escapes_at(text, False, lenient), ESC_OK, text)
        var strict = 0
        assert_equal(
            _escapes_at(text, True, strict),
            ESC_LONE_SURROGATE,
            text,
        )

    # A high surrogate followed by a malformed escape is a hex error
    # either way -- the escape is broken before pairing matters.
    var pos = 0
    assert_equal(_escapes_at("\\uD800\\u1x", False, pos), ESC_BAD_HEX)
    print("  test_escapes_pairs_surrogates passed")


def test_escapes_report_the_position() raises:
    var pos = 0
    _ = _escapes_at("abc\\q", False, pos)
    assert_equal(pos, 3)
    print("  test_escapes_report_the_position passed")


def main() raises:
    print("Whitespace:")
    test_whitespace_set()
    test_skip_ws_scalar_and_vector()
    print()

    print("String classification:")
    test_scan_string_body_flags()
    test_scan_string_body_past_one_vector()
    test_find_control_char()
    print()

    print("UTF-8:")
    test_utf8_accepts_well_formed()
    test_utf8_rejects_malformed()
    test_first_invalid_utf8_points_at_the_byte()
    test_utf8_agrees_with_the_stdlib_constructor()
    print()

    print("Escapes:")
    test_escapes_accepts_the_eight_short_forms()
    test_escapes_rejects_unknown_letters()
    test_escapes_rejects_trailing_backslash()
    test_escapes_validates_hex_digits()
    test_escapes_pairs_surrogates()
    test_escapes_report_the_position()
    print()

    print("All validator tests passed!")
