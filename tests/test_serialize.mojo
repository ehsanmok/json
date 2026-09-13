# Tests for json/serialize.mojo

from std.testing import assert_equal, assert_true, TestSuite

from json import Value, Null, dumps, loads, SerializerConfig


def test_serialize_null() raises:
    """Test serializing null."""
    var v = Value(None)
    assert_equal(dumps(v), "null")


def test_serialize_true() raises:
    """Test serializing true."""
    var v = Value(True)
    assert_equal(dumps(v), "true")


def test_serialize_false() raises:
    """Test serializing false."""
    var v = Value(False)
    assert_equal(dumps(v), "false")


def test_serialize_int_positive() raises:
    """Test serializing positive int."""
    var v = Value(42)
    assert_equal(dumps(v), "42")


def test_serialize_int_negative() raises:
    """Test serializing negative int."""
    var v = Value(-123)
    assert_equal(dumps(v), "-123")


def test_serialize_int_zero() raises:
    """Test serializing zero."""
    var v = Value(0)
    assert_equal(dumps(v), "0")


def test_serialize_string() raises:
    """Test serializing string."""
    var v = Value("hello")
    assert_equal(dumps(v), '"hello"')


def test_serialize_string_empty() raises:
    """Test serializing empty string."""
    var v = Value("")
    assert_equal(dumps(v), '""')


def test_serialize_string_with_escapes() raises:
    """Test serializing string with special characters."""
    var v = Value('hello\nworld\ttab"quote')
    var result = dumps(v)
    assert_equal(result, '"hello\\nworld\\ttab\\"quote"')


def test_dumps_pretty_simple_object() raises:
    """Test pretty printing a simple object."""
    from json import loads

    var data = loads('{"name":"Alice","age":30}')
    var result = dumps(data, indent="  ")
    # Check it contains newlines and indentation
    assert_true(result.find("\n") >= 0, "Should contain newlines")
    assert_true(result.find("  ") >= 0, "Should contain indentation")
    assert_true(result.find('"name"') >= 0, "Should contain name key")
    assert_true(result.find('"Alice"') >= 0, "Should contain Alice value")


def test_dumps_pretty_nested_object() raises:
    """Test pretty printing a nested object."""
    from json import loads

    var data = loads('{"user":{"name":"Bob","scores":[1,2,3]}}')
    var result = dumps(data, indent="  ")
    # Check structure
    assert_true(result.find("\n") >= 0, "Should contain newlines")
    assert_true(result.find('"user"') >= 0, "Should contain user key")
    assert_true(result.find('"name"') >= 0, "Should contain name key")


def test_dumps_pretty_array() raises:
    """Test pretty printing an array."""
    from json import loads

    var data = loads('[1,2,3,"hello",true,null]')
    var result = dumps(data, indent="  ")
    assert_true(result.find("\n") >= 0, "Should contain newlines")
    assert_true(result.find("1") >= 0, "Should contain 1")
    assert_true(result.find('"hello"') >= 0, "Should contain hello")


def test_dumps_pretty_empty_object() raises:
    """Test pretty printing an empty object."""
    from json import loads

    var data = loads("{}")
    var result = dumps(data, indent="  ")
    assert_equal(result, "{}")


def test_dumps_pretty_empty_array() raises:
    """Test pretty printing an empty array."""
    from json import loads

    var data = loads("[]")
    var result = dumps(data, indent="  ")
    assert_equal(result, "[]")


def test_dumps_compact_default() raises:
    """Test that dumps without indent is compact."""
    from json import loads

    var data = loads('{"a":1,"b":2}')
    var result = dumps(data)
    # Should not contain newlines
    assert_true(
        result.find("\n") < 0, "Should not contain newlines in compact mode"
    )


# ---------------------------------------------------------------------------
# Control-character escaping (RFC 8259 s7).
#
# The U+0000..U+001F range MUST be escaped. This had zero coverage, and
# the gap let a real bug ship: escaping was implemented four times over,
# and the copy used for owned-tree containers omitted the range
# entirely, so a control character inside a hand-built object serialized
# to raw bytes -- invalid JSON -- while the identical value read from a
# tape serialized correctly. Our own parser is lenient enough to accept
# the bad output, so a round-trip assertion would not have caught it
# either; these tests assert the exact bytes.
#
# Every case is checked on BOTH representations, since that divergence
# is precisely what went wrong.
# ---------------------------------------------------------------------------


def _ctrl(code: Int) -> String:
    """A one-character string holding the given control byte."""
    return String(chr(code))


def test_escape_control_char_scalar_both_reprs() raises:
    """A bare control character escapes the same either way."""
    var owned = Value("a" + _ctrl(1) + "b")
    assert_equal(dumps(owned), '"a\\u0001b"')
    # Same value, but arriving through the parser (tape-backed).
    var tape = loads('"a\\u0001b"')
    assert_equal(dumps(tape), '"a\\u0001b"')
    assert_equal(dumps(owned), dumps(tape))


def test_escape_control_char_nested_both_reprs() raises:
    """Nested in a container -- this is where the two paths diverged."""
    var built = Value.object()
    built.set("k", Value("a" + _ctrl(1) + "b"))
    assert_equal(dumps(built), '{"k":"a\\u0001b"}')

    var parsed = loads('{"k":"a\\u0001b"}')
    assert_equal(dumps(parsed), '{"k":"a\\u0001b"}')
    assert_equal(dumps(built), dumps(parsed))


def test_escape_control_char_in_array_both_reprs() raises:
    var built = Value.array()
    built.append(Value(_ctrl(31)))
    assert_equal(dumps(built), '["\\u001f"]')
    assert_equal(dumps(loads('["\\u001f"]')), '["\\u001f"]')


def test_escape_control_char_in_key() raises:
    """Object keys need escaping too, not just values."""
    var built = Value.object()
    built.set("a" + _ctrl(2), Value(Int64(1)))
    assert_equal(dumps(built), '{"a\\u0002":1}')
    assert_equal(dumps(loads('{"a\\u0002":1}')), '{"a\\u0002":1}')


def test_escape_backspace_and_formfeed_short_forms() raises:
    """0x08 and 0x0C use the short forms, not \\u0008 / \\u000c."""
    var built = Value.object()
    built.set("k", Value(_ctrl(8) + _ctrl(12)))
    assert_equal(dumps(built), '{"k":"\\b\\f"}')
    assert_equal(dumps(loads('{"k":"\\b\\f"}')), '{"k":"\\b\\f"}')


def test_escape_all_short_forms_together() raises:
    var v = Value.object()
    v.set("k", Value('"' + "\\" + _ctrl(8) + _ctrl(12) + "\n" + "\r" + "\t"))
    assert_equal(dumps(v), '{"k":"\\"\\\\\\b\\f\\n\\r\\t"}')


def test_escape_every_control_byte_both_reprs() raises:
    """Sweep 0x00..0x1F. None may appear raw in the output."""
    for code in range(0, 32):
        var built = Value.object()
        built.set("k", Value(_ctrl(code)))
        var out = dumps(built)
        # No raw control byte survived.
        var b = out.as_bytes()
        for i in range(len(b)):
            assert_true(
                Int(b[i]) >= 0x20,
                String("raw control byte ")
                + String(Int(b[i]))
                + " in output for input code "
                + String(code),
            )
        # And it round-trips back to the same single character.
        var back = loads(out)
        assert_equal(back["k"].string_value().byte_length(), 1)
        assert_equal(Int(back["k"].string_value().as_bytes()[0]), code)


def test_del_byte_is_not_escaped() raises:
    """0x7F is legal raw in a JSON string; the RFC does not require escaping."""
    var built = Value.object()
    built.set("k", Value(_ctrl(0x7F)))
    var out = dumps(built)
    assert_equal(out.byte_length(), 9)  # {"k":"<7f>"}  -> 8 ascii + 1 raw byte
    assert_equal(Int(out.as_bytes()[6]), 0x7F)


def test_non_ascii_passes_through_unescaped() raises:
    """Multi-byte UTF-8 must survive untouched on both paths."""
    var built = Value.object()
    built.set("k", Value("héllo — 日本"))
    var out = dumps(built)
    assert_equal(out, '{"k":"héllo — 日本"}')
    assert_equal(dumps(loads(out)), out)


def test_escaped_output_reparses_to_same_value() raises:
    """Whatever we emit, we must be able to read back identically."""
    var original = "tab\there\nnewline" + _ctrl(1) + 'quote"back\\slash'
    var built = Value.object()
    built.set("k", Value(original))
    var back = loads(dumps(built))
    assert_equal(back["k"].string_value(), original)


def test_config_sorts_object_keys() raises:
    """`sort_keys` used to call a function that returned its input."""
    var v = loads('{"z":1,"a":{"y":2,"b":3},"m":[3,1]}')
    assert_equal(
        dumps(v, SerializerConfig(sort_keys=True)),
        '{"a":{"b":3,"y":2},"m":[3,1],"z":1}',
    )
    # Arrays keep their order; only members are sorted.
    assert_equal(
        dumps(loads("[3,1,2]"), SerializerConfig(sort_keys=True)), "[3,1,2]"
    )


def test_config_escapes_by_code_point_not_by_byte() raises:
    """`escape_unicode` escaped each UTF-8 byte as if it were Latin-1.

    `é` came out as `Ã©`, which is valid JSON that reads back
    as two different characters.
    """
    var v = loads('{"a":"café"}')
    var escaped = dumps(v, SerializerConfig(escape_unicode=True))
    assert_equal(escaped, '{"a":"caf\\u00e9"}')
    assert_equal(dumps(loads(escaped)), dumps(v))


def test_config_escapes_astral_as_a_surrogate_pair() raises:
    var v = loads('["\U0001F600"]')
    var escaped = dumps(v, SerializerConfig(escape_unicode=True))
    assert_equal(escaped, '["\\ud83d\\ude00"]')
    assert_equal(dumps(loads(escaped)), dumps(v))


def test_config_escapes_the_solidus() raises:
    var v = loads('{"a":"x/y"}')
    assert_equal(
        dumps(v, SerializerConfig(escape_forward_slash=True)), '{"a":"x\\/y"}'
    )


def test_config_leaves_non_ascii_alone_when_escaping_slashes() raises:
    """The solidus pass rebuilt the text byte by byte through `chr`."""
    var v = loads('{"a":"café/x"}')
    assert_equal(
        dumps(v, SerializerConfig(escape_forward_slash=True)),
        '{"a":"café\\/x"}',
    )


def test_config_combines_every_option() raises:
    var v = loads('{"z":"a/b","a":"café"}')
    var out = dumps(
        v,
        SerializerConfig(
            indent="  ",
            sort_keys=True,
            escape_unicode=True,
            escape_forward_slash=True,
        ),
    )
    assert_equal(out, '{\n  "a": "caf\\u00e9",\n  "z": "a\\/b"\n}')


def test_ndjson_honours_the_serializer_config() raises:
    """One value per line, so the indent option has nothing to do."""
    var values: List[Value] = [loads('{"z":1,"a":"é"}'), loads('{"b":2}')]
    var out = dumps[format="ndjson"](
        values, SerializerConfig(sort_keys=True, escape_unicode=True)
    )
    assert_equal(out, '{"a":"\\u00e9","z":1}\n{"b":2}')


def main() raises:
    print("=" * 60)
    print("test_serialize.mojo")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
