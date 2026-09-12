# Tests for json/document.mojo
#
# These tests exercise the tape representation in isolation: pack /
# unpack of the bit-packed entries, the side pools, and the builder
# helpers on `Document`. They do NOT depend on `Value` or any parser --
# they verify only that the storage primitive everything else builds
# on is sound.

from std.testing import assert_equal, assert_true, assert_false, TestSuite

from json.document import (
    Document,
    pack_tape_entry,
    tape_tag,
    tape_payload,
    pack_pair,
    payload_hi30,
    payload_lo30,
    TAPE_TAG_NULL,
    TAPE_TAG_BOOL,
    TAPE_TAG_INT,
    TAPE_TAG_FLOAT,
    TAPE_TAG_STRING,
    TAPE_TAG_ARRAY,
    TAPE_TAG_OBJECT,
    TAPE_TAG_KEY,
)


def test_pack_unpack_tag() raises:
    """Tag must round-trip cleanly across all 8 valid tag values."""
    var payload: UInt64 = 0
    for tag in range(8):
        var entry = pack_tape_entry(UInt8(tag), payload)
        assert_equal(Int(tape_tag(entry)), tag, "Tag round-trip")
        assert_equal(Int(tape_payload(entry)), 0, "Empty payload preserved")


def test_pack_unpack_payload_zero() raises:
    """Payload zero round-trips for every tag."""
    var entry = pack_tape_entry(TAPE_TAG_NULL, 0)
    assert_equal(Int(tape_tag(entry)), Int(TAPE_TAG_NULL))
    assert_equal(Int(tape_payload(entry)), 0)


def test_pack_unpack_payload_max60() raises:
    """A 60-bit max payload survives pack/unpack."""
    var max60: UInt64 = (UInt64(1) << 60) - 1
    var entry = pack_tape_entry(TAPE_TAG_INT, max60)
    assert_equal(Int(tape_tag(entry)), Int(TAPE_TAG_INT))
    assert_true(tape_payload(entry) == max60, "Max 60-bit payload preserved")


def test_pack_unpack_pair() raises:
    """Pair packing splits a payload into two 30-bit halves correctly."""
    var hi: UInt64 = 12345
    var lo: UInt64 = 67890
    var packed = pack_pair(hi, lo)
    assert_equal(payload_hi30(packed), 12345)
    assert_equal(payload_lo30(packed), 67890)


def test_pack_unpack_pair_max30() raises:
    """A 30-bit max value survives pair packing."""
    var max30: UInt64 = (UInt64(1) << 30) - 1
    var packed = pack_pair(max30, max30)
    assert_equal(payload_hi30(packed), Int(max30))
    assert_equal(payload_lo30(packed), Int(max30))


def test_document_empty() raises:
    """Default-constructed Document is empty."""
    var d = Document()
    assert_equal(d.size(), 0, "Empty tape")
    assert_equal(len(d.input.as_bytes()), 0, "Empty input")
    assert_equal(len(d.key_pool), 0, "Empty key pool")
    assert_equal(len(d.float_pool), 0, "Empty float pool")


def test_document_with_input() raises:
    """Document remembers the input bytes it was constructed with."""
    var d = Document(String('{"k":1}'))
    assert_equal(d.input, String('{"k":1}'))


def test_append_null() raises:
    """`append_null` produces a NULL entry."""
    var d = Document(String("null"))
    var idx = d.append_null()
    assert_equal(idx, 0)
    assert_equal(d.size(), 1)
    assert_equal(Int(d.get_tag(idx)), Int(TAPE_TAG_NULL))


def test_append_bool_true() raises:
    var d = Document(String("true"))
    var idx = d.append_bool(True)
    assert_equal(Int(d.get_tag(idx)), Int(TAPE_TAG_BOOL))
    assert_true(d.get_bool(idx), "Bool true round-trip")


def test_append_bool_false() raises:
    var d = Document(String("false"))
    var idx = d.append_bool(False)
    assert_equal(Int(d.get_tag(idx)), Int(TAPE_TAG_BOOL))
    assert_false(d.get_bool(idx), "Bool false round-trip")


def test_append_int_small() raises:
    """Small integers round-trip via 60-bit signed encoding."""
    var d = Document()
    var idx = d.append_int(42)
    assert_equal(Int(d.get_tag(idx)), Int(TAPE_TAG_INT))
    assert_equal(Int(d.get_int(idx)), 42)


def test_append_int_negative() raises:
    """Negative integers sign-extend correctly out of the 60-bit field."""
    var d = Document()
    var idx = d.append_int(-12345)
    assert_equal(Int(d.get_int(idx)), -12345)


def test_append_int_minus_one() raises:
    """`-1` is the worst-case sign-extension test."""
    var d = Document()
    var idx = d.append_int(-1)
    assert_equal(Int(d.get_int(idx)), -1)


def test_append_int_zero() raises:
    var d = Document()
    var idx = d.append_int(0)
    assert_equal(Int(d.get_int(idx)), 0)


def test_append_float() raises:
    """Floats spill to float_pool and the tape entry references the offset."""
    var d = Document()
    var idx = d.append_float(3.14159)
    assert_equal(Int(d.get_tag(idx)), Int(TAPE_TAG_FLOAT))
    assert_equal(d.get_float(idx), 3.14159)
    assert_equal(len(d.float_pool), 1)


def test_append_two_floats() raises:
    """Multiple floats accumulate into the pool with stable indices."""
    var d = Document()
    var i1 = d.append_float(1.0)
    var i2 = d.append_float(2.0)
    assert_equal(d.get_float(i1), 1.0)
    assert_equal(d.get_float(i2), 2.0)
    assert_equal(len(d.float_pool), 2)


def test_append_string_slice() raises:
    """STRING entries hold (offset, length) into Document.input."""
    var d = Document(String('"hello world"'))
    # Slice "hello world" lives at offset 1, length 11 (skip the quotes).
    var idx = d.append_string(1, 11)
    assert_equal(Int(d.get_tag(idx)), Int(TAPE_TAG_STRING))
    assert_equal(d.get_string_offset(idx), 1)
    assert_equal(d.get_string_length(idx), 11)
    assert_equal(d.get_string(idx), String("hello world"))


def test_append_key_interns() raises:
    """KEY entries intern unescaped keys into Document.key_pool."""
    var d = Document()
    var idx = d.append_key(String("name"))
    assert_equal(Int(d.get_tag(idx)), Int(TAPE_TAG_KEY))
    assert_equal(d.get_key(idx), String("name"))
    assert_equal(len(d.key_pool), 1)


def test_append_array_layout() raises:
    """ARRAY entries pack (count, child_start_idx) into the payload."""
    var d = Document(String("[1,2]"))
    # Children first, then the parent.
    var c0 = d.append_int(1)
    var c1 = d.append_int(2)
    assert_equal(c0, 0)
    assert_equal(c1, 1)
    var arr = d.append_array(2, c0)
    assert_equal(Int(d.get_tag(arr)), Int(TAPE_TAG_ARRAY))
    assert_equal(d.get_count(arr), 2)
    assert_equal(d.get_child_start(arr), 0)


def test_append_object_layout() raises:
    """OBJECT children are interleaved (key, value) entries."""
    var d = Document(String('{"k":1}'))
    var k0 = d.append_key(String("k"))
    var v0 = d.append_int(1)
    assert_equal(k0, 0)
    assert_equal(v0, 1)
    var obj = d.append_object(1, k0)
    assert_equal(Int(d.get_tag(obj)), Int(TAPE_TAG_OBJECT))
    assert_equal(d.get_count(obj), 1)
    assert_equal(d.get_child_start(obj), 0)


def test_root_points_at_last_entry() raises:
    """`root()` is defined as the last appended entry."""
    var d = Document()
    _ = d.append_null()
    _ = d.append_bool(True)
    var last = d.append_int(7)
    assert_equal(d.root(), last)


def test_multi_view_consistency() raises:
    """Reading the same tape index twice returns the same logical value.

    Two `Value`s pointing at the same `(Document, tape_idx)` must
    observe identical content; this is the foundational property the
    `Value` view relies on.
    """
    var d = Document(String('"foo"'))
    var idx = d.append_string(1, 3)  # "foo"
    var first = d.get_string(idx)
    var second = d.get_string(idx)
    assert_equal(first, second)
    assert_equal(first, String("foo"))


def test_document_copy_is_independent() raises:
    """Document.copy() yields an independent document.

    Mutating one must not affect the other; this guards the
    copy-on-write materialization where an `OwnedValue` clones the
    tape on first write.
    """
    var d1 = Document(String("[1]"))
    _ = d1.append_int(1)
    _ = d1.append_array(1, 0)

    var d2 = d1.copy()
    _ = d1.append_null()  # mutate d1 only

    assert_equal(d1.size(), 3)
    assert_equal(d2.size(), 2)


def main() raises:
    print("=" * 60)
    print("test_document.mojo")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
