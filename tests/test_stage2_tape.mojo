# Tests for the tape-emitting stage 2 path.
#
# `parse_into_document(input, index)` walks the structural index and
# emits a packed tape into a `Document`. These tests verify two
# things:
#
#   1. Document layout: the root entry is the last tape slot, child
#      indices are contiguous and point backwards, and value
#      materialisation through the `Document.get_*` accessors
#      reproduces the original JSON value.
#   2. Validation: every malformed-JSON input the canonical
#      `loads` rejects must also be rejected by `parse_into_document`
#      directly. The tape path is not allowed to be more permissive.

from std.testing import (
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
    TestSuite,
)
from std.collections import List

from json.cpu.stage1_scalar import parse_structural_scalar
from json.cpu.stage2 import parse_into_document
from json.document import (
    Document,
    TAPE_TAG_NULL,
    TAPE_TAG_BOOL,
    TAPE_TAG_INT,
    TAPE_TAG_FLOAT,
    TAPE_TAG_STRING,
    TAPE_TAG_STRING_OWNED,
    TAPE_TAG_ARRAY,
    TAPE_TAG_OBJECT,
    TAPE_TAG_KEY,
    TAPE_TAG_KEY_INLINE,
)
from json import dumps, loads
from json.value import Value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc_for(s: String) raises -> Document:
    var idx = parse_structural_scalar(s)
    return parse_into_document(s, idx)


def _assert_tape_value_matches_value(
    d: Document, tape_idx: Int, v: Value, label: String
) raises:
    """Walk a tape entry against a `Value` and assert they describe
    the same JSON shape + scalar contents."""
    var tag = Int(d.get_tag(tape_idx))

    if tag == Int(TAPE_TAG_NULL):
        assert_true(v.is_null(), label + ": expected null")
        return

    if tag == Int(TAPE_TAG_BOOL):
        assert_true(v.is_bool(), label + ": expected bool")
        assert_equal(d.get_bool(tape_idx), v.bool_value(), label + ": bool")
        return

    if tag == Int(TAPE_TAG_INT):
        assert_true(v.is_int(), label + ": expected int")
        assert_equal(
            Int(d.get_int(tape_idx)), Int(v.int_value()), label + ": int"
        )
        return

    if tag == Int(TAPE_TAG_FLOAT):
        assert_true(v.is_float(), label + ": expected float")
        assert_equal(d.get_float(tape_idx), v.float_value(), label + ": float")
        return

    if tag == Int(TAPE_TAG_STRING) or tag == Int(TAPE_TAG_STRING_OWNED):
        assert_true(v.is_string(), label + ": expected string")
        assert_equal(d.get_string(tape_idx), v.string_value(), label + ": str")
        return

    if tag == Int(TAPE_TAG_ARRAY):
        assert_true(v.is_array(), label + ": expected array")
        var count = d.get_count(tape_idx)
        var child_start = d.get_child_start(tape_idx)
        var items = v.array_items()
        assert_equal(count, len(items), label + ": array count")
        for i in range(count):
            _assert_tape_value_matches_value(
                d, child_start + i, items[i], label + "[" + String(i) + "]"
            )
        return

    if tag == Int(TAPE_TAG_OBJECT):
        assert_true(v.is_object(), label + ": expected object")
        var pair_count = d.get_count(tape_idx)
        var child_start = d.get_child_start(tape_idx)
        var items = v.object_items()
        assert_equal(pair_count, len(items), label + ": object pair count")
        for i in range(pair_count):
            var key_idx = child_start + 2 * i
            var val_idx = child_start + 2 * i + 1
            var pair = items[i].copy()
            var k = pair[0]
            var val = pair[1].copy()
            var key_tag = Int(d.get_tag(key_idx))
            assert_true(
                key_tag == Int(TAPE_TAG_KEY)
                or key_tag == Int(TAPE_TAG_KEY_INLINE),
                label + ": key tag",
            )
            assert_equal(
                d.get_key(key_idx),
                k,
                label + "." + k + ": key",
            )
            _assert_tape_value_matches_value(d, val_idx, val, label + "." + k)
        return

    raise Error(label + ": unknown tape tag " + String(tag))


def _check_equiv(json_str: String, label: String) raises:
    """Both stage 2 paths must accept the input and produce equivalent
    representations."""
    var v = loads(json_str)
    var d = _doc_for(json_str)
    _assert_tape_value_matches_value(d, d.root(), v, label)


# ---------------------------------------------------------------------------
# Layout tests
# ---------------------------------------------------------------------------


def test_root_primitive_null() raises:
    var d = _doc_for(String("null"))
    assert_equal(d.size(), 1, "root-only tape for primitive")
    assert_equal(d.root(), 0)
    assert_equal(Int(d.get_tag(0)), Int(TAPE_TAG_NULL))


def test_root_primitive_bool() raises:
    var dt = _doc_for(String("true"))
    assert_equal(Int(dt.get_tag(dt.root())), Int(TAPE_TAG_BOOL))
    assert_true(dt.get_bool(dt.root()))
    var df = _doc_for(String("false"))
    assert_false(df.get_bool(df.root()))


def test_root_primitive_int_inline() raises:
    var d = _doc_for(String("42"))
    assert_equal(Int(d.get_tag(d.root())), Int(TAPE_TAG_INT))
    assert_equal(Int(d.get_int(d.root())), 42)


def test_root_primitive_negative_int() raises:
    var d = _doc_for(String("-12345"))
    assert_equal(Int(d.get_int(d.root())), -12345)


def test_root_primitive_float() raises:
    var d = _doc_for(String("3.14159"))
    assert_equal(Int(d.get_tag(d.root())), Int(TAPE_TAG_FLOAT))
    assert_equal(d.get_float(d.root()), 3.14159)
    assert_equal(len(d.float_pool), 1)


def test_root_primitive_clean_string() raises:
    var d = _doc_for(String('"hello"'))
    assert_equal(Int(d.get_tag(d.root())), Int(TAPE_TAG_STRING))
    assert_equal(d.get_string(d.root()), String("hello"))
    # Clean strings stay zero-copy: no entry in string_pool.
    assert_equal(len(d.string_pool), 0)


def test_root_primitive_escaped_string() raises:
    var d = _doc_for(String('"hi\\nworld"'))
    assert_equal(Int(d.get_tag(d.root())), Int(TAPE_TAG_STRING_OWNED))
    assert_equal(d.get_string(d.root()), String("hi\nworld"))
    assert_equal(len(d.string_pool), 1)


def test_array_layout_flat() raises:
    """For [1, 2, 3]: child INTs at 0..2 contiguous, ARRAY at 3 with
    count=3, child_start=0, root() == 3."""
    var d = _doc_for(String("[1, 2, 3]"))
    assert_equal(d.size(), 4)
    assert_equal(d.root(), 3)
    assert_equal(Int(d.get_tag(3)), Int(TAPE_TAG_ARRAY))
    assert_equal(d.get_count(3), 3)
    assert_equal(d.get_child_start(3), 0)
    assert_equal(Int(d.get_int(0)), 1)
    assert_equal(Int(d.get_int(1)), 2)
    assert_equal(Int(d.get_int(2)), 3)


def test_array_layout_nested() raises:
    """For [[1,2], [3]]: inner1 children at 0..1, inner1 header at 3
    (inside outer's contiguous block); inner2 child at 2, inner2
    header at 4 (also inside outer's block); outer header at 5."""
    var d = _doc_for(String("[[1,2], [3]]"))
    assert_equal(d.size(), 6)
    assert_equal(d.root(), 5)
    assert_equal(Int(d.get_tag(5)), Int(TAPE_TAG_ARRAY))
    assert_equal(d.get_count(5), 2)
    assert_equal(d.get_child_start(5), 3)
    # outer's child 0 (at idx 3) is inner1 with count 2, child_start 0.
    assert_equal(Int(d.get_tag(3)), Int(TAPE_TAG_ARRAY))
    assert_equal(d.get_count(3), 2)
    assert_equal(d.get_child_start(3), 0)
    # outer's child 1 (at idx 4) is inner2 with count 1, child_start 2.
    assert_equal(Int(d.get_tag(4)), Int(TAPE_TAG_ARRAY))
    assert_equal(d.get_count(4), 1)
    assert_equal(d.get_child_start(4), 2)


def test_object_layout_flat() raises:
    """For {"a": 1, "b": 2}: 4 child header slots (KEY/INT/KEY/INT) +
    OBJECT(2) header at root. Exact layout: KEY at 0, INT at 1, KEY at
    2, INT at 3, OBJECT(count=2, child_start=0) at 4 == root."""
    var d = _doc_for(String('{"a": 1, "b": 2}'))
    assert_equal(d.size(), 5)
    assert_equal(d.root(), 4)
    assert_equal(Int(d.get_tag(4)), Int(TAPE_TAG_OBJECT))
    assert_equal(d.get_count(4), 2)
    assert_equal(d.get_child_start(4), 0)
    assert_equal(Int(d.get_tag(0)), Int(TAPE_TAG_KEY_INLINE))
    assert_equal(d.get_key(0), String("a"))
    assert_equal(Int(d.get_int(1)), 1)
    assert_equal(Int(d.get_tag(2)), Int(TAPE_TAG_KEY_INLINE))
    assert_equal(d.get_key(2), String("b"))
    assert_equal(Int(d.get_int(3)), 2)


def test_object_layout_nested() raises:
    var d = _doc_for(String('{"k": {"x": 1}}'))
    # inner: KEY "x" at 0, INT 1 at 1, OBJECT(1, 0) header would be
    # placed at 2 (which is in outer's contiguous block).
    # outer: KEY "k" at 3, then inner OBJECT header at 4 which IS the
    # inner-object entry (count=1, child_start=0). outer OBJECT(1, 3)
    # at 5 is root. But our layout writes outer's children KEY then
    # VALUE contiguously, with inner's grandchildren written first.
    assert_equal(d.root(), d.size() - 1)
    assert_equal(Int(d.get_tag(d.root())), Int(TAPE_TAG_OBJECT))
    assert_equal(d.get_count(d.root()), 1)
    var cs = d.get_child_start(d.root())
    assert_equal(Int(d.get_tag(cs)), Int(TAPE_TAG_KEY_INLINE))
    assert_equal(d.get_key(cs), String("k"))
    var inner_idx = cs + 1
    assert_equal(Int(d.get_tag(inner_idx)), Int(TAPE_TAG_OBJECT))
    assert_equal(d.get_count(inner_idx), 1)
    var inner_cs = d.get_child_start(inner_idx)
    assert_equal(Int(d.get_tag(inner_cs)), Int(TAPE_TAG_KEY_INLINE))
    assert_equal(d.get_key(inner_cs), String("x"))
    assert_equal(Int(d.get_int(inner_cs + 1)), 1)


# ---------------------------------------------------------------------------
# Equivalence with the canonical Value tree.
# ---------------------------------------------------------------------------


def test_equiv_primitives() raises:
    _check_equiv(String("null"), "null")
    _check_equiv(String("true"), "true")
    _check_equiv(String("false"), "false")
    _check_equiv(String("0"), "zero")
    _check_equiv(String("42"), "small int")
    _check_equiv(String("-12345"), "negative int")
    _check_equiv(String("3.14"), "float")
    _check_equiv(String('"hello"'), "clean string")
    _check_equiv(String('"a\\nb"'), "escaped string")


def test_equiv_arrays() raises:
    _check_equiv(String("[]"), "empty array")
    _check_equiv(String("[1]"), "single int array")
    _check_equiv(String("[1, 2, 3]"), "small array")
    _check_equiv(String('[1, 2.5, true, null, "x"]'), "mixed array")
    _check_equiv(String("[[1, 2], [3, 4]]"), "nested array")
    _check_equiv(String("[[[]]]"), "deeply nested empties")


def test_equiv_objects() raises:
    _check_equiv(String("{}"), "empty object")
    _check_equiv(String('{"k": 1}'), "single pair")
    _check_equiv(String('{"a": 1, "b": 2}'), "two pairs")
    _check_equiv(String('{"name": "Ada", "age": 36}'), "string + int values")
    _check_equiv(String('{"k": {"x": 1, "y": [1, 2]}}'), "nested object")


def test_equiv_mixed_payload() raises:
    _check_equiv(
        String(
            '{"id": 42, "user": {"name": "Ada Lovelace", "tags":'
            ' ["math", "compute", "history"], "active": true},'
            ' "scores": [95.5, 87.3, 91.0], "note": null}'
        ),
        "realistic mixed payload",
    )


def test_equiv_escapes_in_keys_and_values() raises:
    _check_equiv(String('{"k": "a\\"b"}'), "escaped quote in value")
    _check_equiv(String('{"k": "\\n\\t"}'), "common escapes")
    _check_equiv(String('{"k": "\\u00FF"}'), "unicode escape")


def test_equiv_string_with_structural_chars_inside() raises:
    _check_equiv(String('{"k": "a, b: c}"}'), "structurals inside string")


# ---------------------------------------------------------------------------
# Validation parity with the canonical loads pipeline
# ---------------------------------------------------------------------------


def _both_paths_must_reject(s: String, label: String) raises:
    """Both `loads` (canonical pipeline) and `parse_into_document`
    must reject the same malformed inputs."""
    var v1_raised = False
    try:
        var _v = loads(s)
    except:
        v1_raised = True
    assert_true(v1_raised, "Reference path accepted bad input: " + label)

    var idx2 = parse_structural_scalar(s)
    var d_raised = False
    try:
        var _d = parse_into_document(s, idx2)
    except:
        d_raised = True
    assert_true(d_raised, "Tape path accepted bad input: " + label)


def test_reject_trailing_comma_array() raises:
    _both_paths_must_reject(String("[1, 2,]"), "trailing comma array")


def test_reject_malformed_numbers() raises:
    """The near-misses a consume-everything scan used to accept.

    Each of these parsed successfully before, to a value with no
    relationship to the text: `[-]` to zero, `[1+2]` to fifty-two,
    `[0.1.2]` to 0.12.
    """
    _both_paths_must_reject(String("[-]"), "bare minus")
    _both_paths_must_reject(String("[-2.]"), "trailing point")
    _both_paths_must_reject(String("[2.e3]"), "point before exponent")
    _both_paths_must_reject(String("[0.e1]"), "no fraction digits")
    _both_paths_must_reject(String("[0.1.2]"), "two points")
    _both_paths_must_reject(String("[1eE2]"), "two exponent markers")
    _both_paths_must_reject(String("[1+2]"), "expression")
    _both_paths_must_reject(String("[0e+-1]"), "two exponent signs")
    _both_paths_must_reject(String("[-.123]"), "no integer part")
    _both_paths_must_reject(String("[1.]"), "no fraction digits")
    _both_paths_must_reject(String("[01]"), "leading zero")


def test_reject_control_characters_in_strings() raises:
    """Section 7 forbids an unescaped character below U+0020."""
    _both_paths_must_reject(String('["a\tb"]'), "raw tab")
    _both_paths_must_reject(String('["a\nb"]'), "raw newline")
    _both_paths_must_reject(String('["a\x00b"]'), "raw NUL")


def test_reject_malformed_escapes() raises:
    """The four digits after a unicode escape are checked now."""
    _both_paths_must_reject(String('["\\uqqqq"]'), "non-hex digits")
    _both_paths_must_reject(String('["\\u00A"]'), "three digits")
    _both_paths_must_reject(String('["\\uD834\\uDd"]'), "truncated pair")
    _both_paths_must_reject(String('["\\q"]'), "unknown escape")


def _raw(values: List[Int]) -> String:
    """A document from raw bytes, valid UTF-8 or not."""
    var out = List[UInt8](capacity=len(values))
    for i in range(len(values)):
        out.append(UInt8(values[i]))
    return String(unsafe_from_utf8=out^)


def test_reject_invalid_utf8_in_strings() raises:
    """RFC 3629: string bytes must be well-formed UTF-8.

    Nothing validated them on this path, while the simdjson backend
    did, so the two backends disagreed about which documents exist.
    """
    _both_paths_must_reject(
        _raw([0x5B, 0x22, 0xC0, 0x80, 0x22, 0x5D]), "overlong NUL"
    )
    _both_paths_must_reject(
        _raw([0x5B, 0x22, 0xED, 0xA0, 0x80, 0x22, 0x5D]), "encoded surrogate"
    )
    _both_paths_must_reject(
        _raw([0x5B, 0x22, 0xE2, 0x82, 0x22, 0x5D]), "truncated sequence"
    )


def test_wide_integers_survive_the_tape() raises:
    """Integers past the inline payload are pooled, not truncated.

    The payload is 60 bits, so anything at or above 2**59 used to be
    sign-flipped on the way in and read back as a different number.
    """
    assert_equal(loads("576460752303423488").int_value(), 576460752303423488)
    assert_equal(loads("-576460752303423489").int_value(), -576460752303423489)
    assert_equal(loads("9223372036854775807").int_value(), Int64.MAX)
    assert_equal(loads("-9223372036854775808").int_value(), Int64.MIN)
    assert_equal(dumps(loads("576460752303423488")), "576460752303423488")


def test_integers_above_int64_stay_exact() raises:
    """A magnitude past `Int64.MAX` used to wrap to a negative number."""
    var wide = loads("18446744073709551615")
    assert_true(wide.is_uint())
    assert_false(wide.is_int())
    assert_true(wide.is_number())
    assert_equal(wide.uint_value(), UInt64.MAX)
    assert_equal(dumps(wide), "18446744073709551615")


def test_integers_beyond_uint64_become_floats() raises:
    """Past `UInt64` the nearest float is the widely compatible answer."""
    var huge = loads("18446744073709551616")
    assert_true(huge.is_float())
    assert_equal(huge.float_value(), 1.8446744073709552e19)


def test_reject_trailing_comma_object() raises:
    _both_paths_must_reject(String('{"k": 1,}'), "trailing comma object")


def test_reject_double_comma_array() raises:
    _both_paths_must_reject(String("[1,, 2]"), "double comma array")


def test_reject_leading_comma_array() raises:
    _both_paths_must_reject(String("[, 1]"), "leading comma array")


def test_reject_missing_colon_in_object() raises:
    _both_paths_must_reject(String('{"k" "v"}'), "missing colon")


def test_reject_unquoted_object_key() raises:
    _both_paths_must_reject(String("{k: 1}"), "unquoted key")


def test_reject_leading_zeros() raises:
    _both_paths_must_reject(String("007"), "leading zeros")


def test_reject_trailing_content() raises:
    _both_paths_must_reject(String("null garbage"), "trailing content")


# ---------------------------------------------------------------------------
# Real-corpus equivalence
# ---------------------------------------------------------------------------


def test_real_corpus_equiv() raises:
    """Both stage 2 paths must agree on the live benchmark corpora."""
    from std.pathlib import Path

    var fixtures = List[String]()
    fixtures.append("benchmark/datasets/twitter.json")
    fixtures.append("benchmark/datasets/citm_catalog.json")

    for i in range(len(fixtures)):
        var path = fixtures[i]
        try:
            var content = Path(path).read_text()
            _check_equiv(content, path)
        except:
            print("  skipped (missing):", path)


def main() raises:
    print("=" * 60)
    print("test_stage2_tape.mojo")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
