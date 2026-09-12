"""Tests for the byte cursor behind typed deserialization.

The string scanner is the part worth pinning. It finds the closing
quote and classifies the body in one vector pass, which means a block
can contain both the end of this string and the start of whatever
follows it. If the flags were taken from the whole block rather than
from the bytes before the quote, a control character or a non-ASCII
byte in the *next* token would be attributed to this one and a valid
document would be rejected. Several tests below exist only to hold
that line.
"""

from std.collections import List
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from json import loads
from json.reader import JsonReader


def _read_one(text: String) raises -> String:
    var r = JsonReader(text.as_bytes())
    return r.read_string()


def _raw(values: List[Int]) -> String:
    var out = List[UInt8](capacity=len(values))
    for i in range(len(values)):
        out.append(UInt8(values[i]))
    return String(unsafe_from_utf8=out^)


# ===================================================================
# Strings
# ===================================================================


def test_reads_plain_strings() raises:
    assert_equal(_read_one('"abc"'), "abc")
    assert_equal(_read_one('""'), "")
    assert_equal(
        _read_one('"a longer string past one vector block"'),
        "a longer string past one vector block",
    )
    print("  test_reads_plain_strings passed")


def test_quote_at_every_block_offset() raises:
    """The closing quote landing anywhere in or across a block.

    Sixteen bytes is one register here, so offsets on either side of a
    block boundary are where a fused scan goes wrong.
    """
    for length in range(0, 40):
        var body = String("a" * length)
        assert_equal(
            _read_one('"' + body + '"'), body, "length " + String(length)
        )
    print("  test_quote_at_every_block_offset passed")


def test_bytes_after_the_quote_do_not_leak_into_flags() raises:
    """A control byte or non-ASCII byte after the quote is not ours.

    Both of these are valid documents. If the scan took its flags from
    the whole block rather than from the bytes before the closing
    quote, the first would be rejected as a control character in a
    string and the second would drag the following text into a UTF-8
    check for a string that is pure ASCII.
    """
    # A raw tab between two members: legal whitespace, illegal inside a
    # string. It sits in the same 16-byte block as the first value.
    var with_tab = String('{"k":"ab",\t"j":"cd"}')
    var parsed = loads(with_tab)
    assert_equal(parsed["k"].string_value(), "ab")
    assert_equal(parsed["j"].string_value(), "cd")

    # Non-ASCII in the *next* string, close enough to share a block.
    var with_utf8 = String('{"k":"ab","j":"café"}')
    var parsed2 = loads(with_utf8)
    assert_equal(parsed2["k"].string_value(), "ab")
    assert_equal(parsed2["j"].string_value(), "café")

    # Same, through the reader directly: read one short string whose
    # block reaches into a following newline and a following é.
    var direct = String('"ab"\n"café"')
    var r = JsonReader(direct.as_bytes())
    assert_equal(r.read_string(), "ab")
    assert_equal(r.read_string(), "café")
    print("  test_bytes_after_the_quote_do_not_leak_into_flags passed")


def test_escapes_across_a_block_boundary() raises:
    """An escape that starts in one block and ends in the next."""
    for pad in range(0, 20):
        var text = '"' + String("a" * pad) + "\\n" + 'z"'
        var want = String("a" * pad) + "\n" + "z"
        assert_equal(_read_one(text), want, "pad " + String(pad))
    # An escaped quote must not end the string, at any offset.
    for pad in range(0, 20):
        var text = '"' + String("a" * pad) + '\\"' + 'z"'
        var want = String("a" * pad) + '"' + "z"
        assert_equal(_read_one(text), want, "pad " + String(pad))
    print("  test_escapes_across_a_block_boundary passed")


def test_unicode_escapes_and_surrogates() raises:
    assert_equal(_read_one('"\\u0041"'), "A")
    assert_equal(_read_one('"\\uD834\\uDD1E"'), "\U0001D11E")
    assert_equal(_read_one('"a\\u00e9b"'), "aéb")
    print("  test_unicode_escapes_and_surrogates passed")


def test_rejects_bad_strings() raises:
    """The same rules the tape parser enforces, from the same helpers."""
    for text in [
        '"unterminated',
        '"a\tb"',
        '"a\nb"',
        '"\\q"',
        '"\\uqqqq"',
        '"\\u00A"',
        '"abc\\"',
    ]:
        with assert_raises():
            _ = _read_one(text)
    # Invalid UTF-8 inside a string.
    with assert_raises(contains="UTF-8"):
        _ = _read_one(_raw([0x22, 0xC0, 0x80, 0x22]))
    print("  test_rejects_bad_strings passed")


def test_long_strings_keep_their_flags() raises:
    """A body longer than one block still reports what it contains."""
    var long_plain = String("x" * 100)
    assert_equal(_read_one('"' + long_plain + '"'), long_plain)

    var late_escape = String("x" * 100) + "\\n"
    assert_equal(_read_one('"' + late_escape + '"'), String("x" * 100) + "\n")

    with assert_raises():
        _ = _read_one('"' + String("x" * 100) + "\t" + '"')
    print("  test_long_strings_keep_their_flags passed")


# ===================================================================
# Agreement with the tape parser
# ===================================================================


def test_agrees_with_loads_on_the_conformance_corpus() raises:
    """Two readers, one grammar.

    `loads` builds a document and the reader walks bytes into typed
    values, but they share the number scanner and the string
    validators, so they must accept and reject exactly the same
    inputs. Anything else means one of them has its own idea of what
    JSON is.
    """
    from std.pathlib import Path

    var catalog = loads(Path("tests/conformance/rfc8259.json").read_text())
    var cases = catalog["cases"].array_items()
    var checked = 0
    var disagreements = 0
    for i in range(len(cases)):
        ref item = cases[i]
        var encoding = "utf-8"
        var keys = item.object_keys()
        for k in range(len(keys)):
            if keys[k] == "input_encoding":
                encoding = item["input_encoding"].string_value()
        if encoding != "utf-8":
            continue
        var text = item["input"].string_value()

        var tape_ok = True
        try:
            _ = loads(text)
        except:
            tape_ok = False

        var reader_ok = True
        try:
            var r = JsonReader(text.as_bytes())
            _ = r.skip_value()
            r.expect_end()
        except:
            reader_ok = False

        checked += 1
        if tape_ok != reader_ok:
            disagreements += 1
            if disagreements < 6:
                print(
                    "    disagreement:",
                    item["id"].string_value(),
                    "tape",
                    tape_ok,
                    "reader",
                    reader_ok,
                )
    assert_true(checked > 300, "expected the whole catalog")
    assert_equal(disagreements, 0)
    print("  test_agrees_with_loads_on_the_conformance_corpus passed")


def test_depth_limit() raises:
    var deep = String("[" * 1025 + "]" * 1025)
    var r = JsonReader(deep.as_bytes())
    with assert_raises(contains="nesting depth"):
        _ = r.skip_value()

    var fine = String("[" * 1000 + "]" * 1000)
    var r2 = JsonReader(fine.as_bytes())
    _ = r2.skip_value()
    print("  test_depth_limit passed")


def main() raises:
    print("Strings:")
    test_reads_plain_strings()
    test_quote_at_every_block_offset()
    test_bytes_after_the_quote_do_not_leak_into_flags()
    test_escapes_across_a_block_boundary()
    test_unicode_escapes_and_surrogates()
    test_rejects_bad_strings()
    test_long_strings_keep_their_flags()
    print()

    print("Agreement:")
    test_agrees_with_loads_on_the_conformance_corpus()
    test_depth_limit()
    print()

    print("All reader tests passed!")
