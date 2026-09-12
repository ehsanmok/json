# Tests for json/writer.mojo -- the byte-level serialization sink.
#
# Two things need pinning here. First the output bytes, since every
# serialization path in the library now emits through this type. Second
# the buffer mechanics: `ensure()` growing a short buffer, and `finish()`
# trimming to `pos` rather than returning the whole allocation. Those are
# easy to get subtly wrong and invisible in the happy path, because a
# generous capacity estimate hides them.

from std.testing import assert_equal, assert_true, assert_false, TestSuite

from json.writer import (
    _DIGIT_PAIRS,
    _DIGIT_PAIRS_ARRAY,
    _digit_pair,
    JsonWriter,
    needs_escape,
)


def test_scalars() raises:
    var w = JsonWriter(capacity=64)
    w.write_null()
    assert_equal(w^.finish_string(), "null")

    var t = JsonWriter(capacity=64)
    t.write_bool(True)
    assert_equal(t^.finish_string(), "true")

    var f = JsonWriter(capacity=64)
    f.write_bool(False)
    assert_equal(f^.finish_string(), "false")


def test_int_boundaries() raises:
    """Digit-count ladder and the backwards pair writer."""
    var cases = [
        (Int64(0), String("0")),
        (Int64(7), String("7")),
        (Int64(-7), String("-7")),
        (Int64(10), String("10")),
        (Int64(99), String("99")),
        (Int64(100), String("100")),
        (Int64(101), String("101")),
        (Int64(-101), String("-101")),
        (Int64(999), String("999")),
        (Int64(1000), String("1000")),
        (Int64(999999999), String("999999999")),
        (Int64(1000000000), String("1000000000")),
        (Int64(9999999999), String("9999999999")),
        (Int64(10000000000), String("10000000000")),
        # Past 2^53, where a float-based formatter would lose precision.
        (Int64(9007199254740993), String("9007199254740993")),
        (Int64(9223372036854775807), String("9223372036854775807")),
        (Int64.MIN, String("-9223372036854775808")),
    ]
    for c in cases:
        var w = JsonWriter(capacity=32)
        w.write_int(c[0])
        assert_equal(w^.finish_string(), c[1])


def test_string_clean_fast_path() raises:
    var w = JsonWriter(capacity=64)
    w.write_string("hello world")
    assert_equal(w^.finish_string(), '"hello world"')


def test_string_empty() raises:
    var w = JsonWriter(capacity=8)
    w.write_string("")
    assert_equal(w^.finish_string(), '""')


def test_string_escapes() raises:
    var w = JsonWriter(capacity=8)
    w.write_string('a"b\\c')
    assert_equal(w^.finish_string(), '"a\\"b\\\\c"')

    var c = JsonWriter(capacity=8)
    c.write_string("\n\r\t")
    assert_equal(c^.finish_string(), '"\\n\\r\\t"')


def test_string_control_bytes_hex_escaped() raises:
    """C0 controls without a short form take the six-character escape."""
    var w = JsonWriter(capacity=8)
    w.write_string(String(chr(1)) + String(chr(31)))
    assert_equal(w^.finish_string(), '"\\u0001\\u001f"')


def test_string_escape_at_simd_boundary() raises:
    """An escape past the first SIMD chunk must still be found.

    The scan is chunked, so a string long enough to have a clean first
    chunk and a dirty later one exercises the run-copy path rather than
    the memcpy fast path.
    """
    var s = String()
    for _ in range(200):
        s += "a"
    s += '"'
    for _ in range(200):
        s += "b"
    var w = JsonWriter(capacity=16)
    w.write_string(s)
    var out = w^.finish_string()
    # 401 input chars, one of which becomes two, plus two quotes.
    assert_equal(out.byte_length(), 404)
    assert_true('a\\"b' in out)


def test_needs_escape_detection() raises:
    assert_false(needs_escape(String("plain text").as_bytes()))
    assert_false(needs_escape(String("").as_bytes()))
    assert_true(needs_escape(String('has "quote"').as_bytes()))
    assert_true(needs_escape(String("back\\slash").as_bytes()))
    assert_true(needs_escape(String("nl\n").as_bytes()))
    # Long clean string: forces the SIMD loop, not just the tail.
    var long = String()
    for _ in range(500):
        long += "x"
    assert_false(needs_escape(long.as_bytes()))
    assert_true(needs_escape((long + "\t").as_bytes()))


def test_non_ascii_passes_through() raises:
    var w = JsonWriter(capacity=8)
    w.write_string("héllo 日本")
    assert_equal(w^.finish_string(), '"héllo 日本"')


def test_object_and_array_separators() raises:
    """Comma bookkeeping is the writer's job, not the caller's."""
    var w = JsonWriter(capacity=64)
    w.begin_object()
    w.key("a")
    w.write_int(1)
    w.value_written()
    w.key("b")
    w.write_string("x")
    w.value_written()
    w.end_object()
    assert_equal(w^.finish_string(), '{"a":1,"b":"x"}')


def test_array_items() raises:
    var w = JsonWriter(capacity=64)
    w.begin_array()
    w.item_int(1)
    w.item_int(2)
    w.item_string("three")
    w.item_bool(True)
    w.item_null()
    w.end_array()
    assert_equal(w^.finish_string(), '[1,2,"three",true,null]')


def test_empty_containers() raises:
    var o = JsonWriter(capacity=16)
    o.begin_object()
    o.end_object()
    assert_equal(o^.finish_string(), "{}")

    var a = JsonWriter(capacity=16)
    a.begin_array()
    a.end_array()
    assert_equal(a^.finish_string(), "[]")


def test_nested_containers() raises:
    var w = JsonWriter(capacity=16)
    w.begin_object()
    w.key("outer")
    w.begin_array()
    w.begin_object()
    w.key("k")
    w.write_int(1)
    w.value_written()
    w.end_object()
    w.begin_object()
    w.key("k")
    w.write_int(2)
    w.value_written()
    w.end_object()
    w.end_array()
    w.end_object()
    assert_equal(w^.finish_string(), '{"outer":[{"k":1},{"k":2}]}')


def test_ensure_grows_from_tiny_capacity() raises:
    """A deliberately hopeless estimate must still produce correct bytes."""
    var w = JsonWriter(capacity=1)
    w.begin_array()
    for i in range(200):
        w.item_int(Int64(i))
    w.end_array()
    var out = w^.finish_string()
    assert_true(out.startswith("[0,1,2,"))
    assert_true(out.endswith(",198,199]"))


def test_zero_capacity() raises:
    var w = JsonWriter(capacity=0)
    w.write_string("still works")
    assert_equal(w^.finish_string(), '"still works"')


def test_finish_trims_to_written_length() raises:
    """The finish() result is `pos` bytes long, not the whole allocation."""
    var w = JsonWriter(capacity=4096)
    w.write_int(42)
    var bytes = w^.finish()
    assert_equal(len(bytes), 2)
    assert_equal(Int(bytes[0]), ord("4"))
    assert_equal(Int(bytes[1]), ord("2"))


def test_adopted_buffer_is_reused() raises:
    """The buffer-adopting constructor allows allocation reuse."""
    var first = JsonWriter(capacity=128)
    first.write_int(1)
    var buf = first^.finish()
    var reused = JsonWriter(buf^)
    reused.write_string("second")
    assert_equal(reused^.finish_string(), '"second"')


def test_float_round_trips() raises:
    var w = JsonWriter(capacity=32)
    w.write_float(1.5)
    assert_equal(w^.finish_string(), "1.5")

    var z = JsonWriter(capacity=32)
    z.write_float(0.0)
    assert_true(z^.finish_string().startswith("0"))


def test_digit_tables_agree() raises:
    """The text and array forms of the digit table hold the same digits.

    The integer writer reads the text and the exponent writer reads
    the array, each because it measured faster there. Nothing in the
    build checks that the two were written the same way, so this does.
    """
    assert_equal(_DIGIT_PAIRS.byte_length(), 200)
    var array = materialize[_DIGIT_PAIRS_ARRAY]()
    for value in range(100):
        var text = _digit_pair(value)
        assert_equal(Int(text[0]), 0x30 + value // 10)
        assert_equal(Int(text[1]), 0x30 + value % 10)
        assert_equal(Int(array[value][0]), Int(text[0]))
        assert_equal(Int(array[value][1]), Int(text[1]))


def main() raises:
    print("=" * 60)
    print("test_writer.mojo")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
