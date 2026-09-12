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
from json.reflection import deserialize_json


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
# Numbers
# ===================================================================


def _read_i64(text: String) raises -> Int64:
    var r = JsonReader(text.as_bytes())
    return r.read_int[DType.int64]()


def _read_i32(text: String) raises -> Int32:
    var r = JsonReader(text.as_bytes())
    return r.read_int[DType.int32]()


def test_integer_fast_path_values() raises:
    """The short-integer path and the scanner must agree on values."""
    assert_equal(_read_i64("0"), 0)
    assert_equal(_read_i64("-0"), 0)
    assert_equal(_read_i64("7"), 7)
    assert_equal(_read_i64("-7"), -7)
    assert_equal(_read_i64("12345"), 12345)
    assert_equal(_read_i64("999999999999999999"), 999999999999999999)
    assert_equal(_read_i64("-999999999999999999"), -999999999999999999)
    # Nineteen digits and up leave the fast path to the scanner.
    assert_equal(_read_i64("9223372036854775807"), Int64.MAX)
    assert_equal(_read_i64("-9223372036854775808"), Int64.MIN)
    print("  test_integer_fast_path_values passed")


def test_integer_fast_path_stops_at_the_right_byte() raises:
    """A number ends where the grammar says, not where digits stop."""
    var r = JsonReader(String("[12,34]").as_bytes())
    r.expect_array_begin()
    var first = True
    _ = r.next_element(first)
    assert_equal(r.read_int[DType.int64](), 12)
    _ = r.next_element(False)
    assert_equal(r.read_int[DType.int64](), 34)
    assert_false(r.next_element(False))
    print("  test_integer_fast_path_stops_at_the_right_byte passed")


def test_integer_fast_path_defers_on_everything_it_cannot_prove() raises:
    """Malformed numbers still raise, with the scanner's messages.

    The fast path exists only to skip work, never to decide. Anything
    it cannot fully establish -- a leading zero, a fraction, an
    exponent, a bare minus -- has to reach the scanner, or the two
    readers would disagree about what a number is.
    """
    for text in ["01", "-01", "1.", "-", "1e", "1e+", ".5", "+1", "1.2.3"]:
        with assert_raises():
            _ = _read_i64(text)
    # A fraction or exponent is a type error, not a truncation.
    with assert_raises(contains="expected an integer"):
        _ = _read_i64("1.5")
    with assert_raises(contains="expected an integer"):
        _ = _read_i64("1e3")
    print(
        "  test_integer_fast_path_defers_on_everything_it_cannot_prove passed"
    )


def test_integer_range_errors_match_the_scanner() raises:
    """Both paths report an out-of-range integer the same way."""
    assert_equal(_read_i32("2147483647"), Int32.MAX)
    assert_equal(_read_i32("-2147483648"), Int32.MIN)
    with assert_raises(contains="out of range"):
        _ = _read_i32("2147483648")
    with assert_raises(contains="out of range"):
        _ = _read_i32("-2147483649")
    # Past Int64 entirely: the scanner's territory.
    with assert_raises(contains="out of range"):
        _ = _read_i64("9223372036854775808")
    print("  test_integer_range_errors_match_the_scanner passed")


def test_unsigned_fields_reject_negatives() raises:
    var r = JsonReader(String("-1").as_bytes())
    with assert_raises(contains="unsigned"):
        _ = r.read_int[DType.uint32]()
    var r2 = JsonReader(String("4294967295").as_bytes())
    assert_equal(r2.read_int[DType.uint32](), UInt32.MAX)
    print("  test_unsigned_fields_reject_negatives passed")


# ===================================================================
# Keys
# ===================================================================


@fieldwise_init
struct Row(Movable):
    var sku: String
    var qty: Int64


@fieldwise_init
struct One(Movable):
    var value: Int64


def test_key_matching_is_exact() raises:
    """A key matches a field name only when it is that name.

    The comparison reads the key's bytes out of the document, so a
    name that is a prefix of the key -- or the other way round -- must
    be settled by the length test rather than by running off the end
    of one into whatever follows it.
    """
    var exact = deserialize_json[Row]('{"sku":"a","qty":2}')
    assert_equal(exact.sku, "a")
    assert_equal(exact.qty, 2)

    # Keys that share a prefix with a field name, in both directions,
    # are unknown fields and are skipped -- not matched.
    with assert_raises(contains="missing required field 'sku'"):
        _ = deserialize_json[Row]('{"sk":"a","qty":2}')
    with assert_raises(contains="missing required field 'sku'"):
        _ = deserialize_json[Row]('{"sku_":"a","qty":2}')
    with assert_raises(contains="missing required field 'qty'"):
        _ = deserialize_json[Row]('{"sku":"a","qtyx":2}')

    # An escaped key still matches the name it spells.
    var escaped = deserialize_json[Row]('{"\u0073ku":"a","qty":2}')
    assert_equal(escaped.sku, "a")
    print("  test_key_matching_is_exact passed")


def test_key_at_the_end_of_the_document() raises:
    """A key comparison must not read past the document.

    The name is known at compile time and the key's length is known
    from the scan, so a name longer than the key is settled without a
    read -- which is what keeps a key at the very end of the buffer
    from being compared against bytes that are not there.
    """
    with assert_raises():
        _ = deserialize_json[One]('{"v"')
    var ok = deserialize_json[One]('{"value":1}')
    assert_equal(ok.value, 1)
    print("  test_key_at_the_end_of_the_document passed")


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

    print("Numbers:")
    test_integer_fast_path_values()
    test_integer_fast_path_stops_at_the_right_byte()
    test_integer_fast_path_defers_on_everything_it_cannot_prove()
    test_integer_range_errors_match_the_scanner()
    test_unsigned_fields_reject_negatives()
    print()

    print("Keys:")
    test_key_matching_is_exact()
    test_key_at_the_end_of_the_document()
    print()

    print("Agreement:")
    test_agrees_with_loads_on_the_conformance_corpus()
    test_depth_limit()
    print()

    print("All reader tests passed!")
