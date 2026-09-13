# Comprehensive tests for the unified API

from std.os import remove
from std.testing import assert_equal, assert_true, TestSuite

from json import (
    loads,
    dumps,
    load,
    dump,
    Value,
    ParserConfig,
    SerializerConfig,
    LazyValue,
    StreamingParser,
    apply_patch,
    merge_patch,
    jsonpath_query,
    validate,
    is_valid,
)


# =============================================================================
# loads() tests
# =============================================================================


def test_loads_basic() raises:
    """Test basic JSON parsing."""
    var data = loads('{"name":"Alice","age":30}')
    assert_equal(data["name"].string_value(), "Alice")
    assert_equal(Int(data["age"].int_value()), 30)


def test_loads_with_config() raises:
    """Test loads with ParserConfig."""
    var config = ParserConfig(allow_comments=True, allow_trailing_comma=True)
    var data = loads('{"a": 1,} // comment', config)
    assert_equal(Int(data["a"].int_value()), 1)


def test_loads_ndjson() raises:
    """Test loads with format=ndjson."""
    var values = loads[format="ndjson"]('{"a":1}\n{"a":2}\n{"a":3}')
    assert_equal(len(values), 3)
    assert_equal(Int(values[0]["a"].int_value()), 1)
    assert_equal(Int(values[2]["a"].int_value()), 3)


def test_loads_lazy() raises:
    """Test loads with lazy=True."""
    var lazy = loads[lazy=True]('{"users":[{"name":"Alice"},{"name":"Bob"}]}')
    var name = lazy.get("/users/0/name")
    assert_equal(name.string_value(), "Alice")


def test_loads_all_types() raises:
    """Test parsing all JSON types."""
    assert_true(loads("null").is_null())
    assert_true(loads("true").is_bool())
    assert_true(loads("false").is_bool())
    assert_true(loads("42").is_int())
    assert_true(loads("3.14").is_float())
    assert_true(loads('"hello"').is_string())
    assert_true(loads("[1,2,3]").is_array())
    assert_true(loads('{"a":1}').is_object())


# =============================================================================
# dumps() tests
# =============================================================================


def test_dumps_basic() raises:
    """Test basic serialization."""
    var data = loads('{"a":1}')
    var s = dumps(data)
    assert_true(s.find('"a"') >= 0)
    assert_true(s.find("1") >= 0)


def test_dumps_pretty() raises:
    """Test dumps with indentation."""
    var data = loads('{"a":1,"b":2}')
    var s = dumps(data, indent="  ")
    assert_true(s.find("\n") >= 0)


def test_dumps_config() raises:
    """Test dumps with SerializerConfig."""
    var data = loads('{"url":"http://x.com"}')
    var config = SerializerConfig(escape_forward_slash=True)
    var s = dumps(data, config)
    assert_true(s.find("\\/") >= 0)


def test_dumps_ndjson() raises:
    """Test dumps with format=ndjson."""
    var values = List[Value]()
    values.append(loads('{"a":1}'))
    values.append(loads('{"a":2}'))
    var s = dumps[format="ndjson"](values)
    assert_true(s.find("\n") >= 0)


def test_dumps_roundtrip() raises:
    """Test serialization roundtrip."""
    var original = '{"name":"Alice","scores":[95,87,92]}'
    var data = loads(original)
    var serialized = dumps(data)
    var reparsed = loads(serialized)
    assert_equal(reparsed["name"].string_value(), "Alice")


# =============================================================================
# load()/dump() file tests
# =============================================================================


def test_load_dump_roundtrip() raises:
    """Test file load/dump roundtrip."""
    var data = loads('{"test":123,"arr":[1,2,3]}')

    var f_out = open("test_api.json", "w")
    dump(data, f_out)
    f_out.close()

    var loaded = load("test_api.json")
    assert_equal(Int(loaded["test"].int_value()), 123)


def test_load_ndjson() raises:
    """Test load auto-detects .ndjson files."""
    var f_out = open("test_api.ndjson", "w")
    f_out.write('{"a":1}\n{"a":2}\n')
    f_out.close()

    var data = load("test_api.ndjson")  # Auto-detects from extension
    assert_true(data.is_array())
    assert_equal(data.array_count(), 2)


def test_load_streaming() raises:
    """Test load with streaming=True."""
    var f_out = open("test_api_stream.ndjson", "w")
    f_out.write('{"a":1}\n{"a":2}\n{"a":3}\n')
    f_out.close()

    var parser = load[streaming=True]("test_api_stream.ndjson")
    var count = 0
    while parser.has_next():
        _ = parser.next()
        count += 1
    parser.close()

    assert_equal(count, 3)


# =============================================================================
# Value operations
# =============================================================================


def test_value_access() raises:
    """Test value access methods."""
    var data = loads('{"users":[{"name":"Alice"},{"name":"Bob"}]}')
    assert_equal(data["users"][0]["name"].string_value(), "Alice")
    assert_equal(data.at("/users/1/name").string_value(), "Bob")


def test_value_mutation() raises:
    """Test value mutation."""
    var data = loads('{"a":1}')
    data.set("b", Value(2))
    assert_equal(Int(data["b"].int_value()), 2)


def test_value_iteration() raises:
    """Test array/object iteration."""
    var arr = loads("[1,2,3]")
    var items = arr.array_items()
    assert_equal(len(items), 3)

    var obj = loads('{"a":1,"b":2}')
    var pairs = obj.object_items()
    assert_equal(len(pairs), 2)


# =============================================================================
# Advanced features
# =============================================================================


def test_jsonpath() raises:
    """Test JSONPath queries."""
    var data = loads('{"users":[{"name":"Alice"},{"name":"Bob"}]}')
    var names = jsonpath_query(data, "$.users[*].name")
    assert_equal(len(names), 2)


def test_json_patch() raises:
    """Test JSON Patch."""
    var doc = loads('{"a":1}')
    var patch = loads('[{"op":"add","path":"/b","value":2}]')
    var result = apply_patch(doc, patch)
    assert_equal(Int(result["b"].int_value()), 2)


def test_merge_patch() raises:
    """Test JSON Merge Patch."""
    var target = loads('{"a":1,"b":2}')
    var patch = loads('{"b":null,"c":3}')
    var result = merge_patch(target, patch)
    assert_equal(Int(result["c"].int_value()), 3)


def test_schema_validation() raises:
    """Test JSON Schema validation."""
    var schema = loads('{"type":"object","required":["name"]}')
    var valid_doc = loads('{"name":"Alice"}')
    var invalid_doc = loads('{"age":30}')

    assert_true(is_valid(valid_doc, schema))
    assert_true(not is_valid(invalid_doc, schema))


# =============================================================================
# NDJSON roundtrip
# =============================================================================


def test_ndjson_roundtrip() raises:
    """Test NDJSON roundtrip."""
    var original = '{"id":1}\n{"id":2}\n{"id":3}'
    var values = loads[format="ndjson"](original)
    var serialized = dumps[format="ndjson"](values)
    var reparsed = loads[format="ndjson"](serialized)
    assert_equal(len(reparsed), 3)


def test_parser_config_keeps_non_ascii_intact() raises:
    """Preprocessing rebuilt the document through `chr` per byte.

    Any byte above 0x7F was read as a code point and written back as
    two, so a document with non-ASCII text was corrupted before the
    parser saw it. This only happened when a `ParserConfig` asked for
    comments or trailing commas, which is exactly when nobody looks.
    """
    var config = ParserConfig(allow_comments=True, allow_trailing_comma=True)
    var data = loads('{"a": "café 😀" /* note */, "b": [1, 2, ],}', config)
    assert_equal(data["a"].string_value(), "café 😀")
    assert_equal(data["b"].array_count(), 2)


def test_parser_config_leaves_string_contents_alone() raises:
    """A comma inside a string is not a trailing comma."""
    var config = ParserConfig(allow_trailing_comma=True)
    var data = loads('{"a": "x, }", "b": [1,]}', config)
    assert_equal(data["a"].string_value(), "x, }")
    assert_equal(data["b"].array_count(), 1)


def test_parser_config_enforces_max_depth() raises:
    var config = ParserConfig(max_depth=3)
    _ = loads("[[1]]", config)
    var raised = False
    try:
        _ = loads("[[[[1]]]]", config)
    except:
        raised = True
    assert_true(raised)


def test_ijson_rejects_duplicate_member_names() raises:
    """RFC 7493 section 2.1. The default parser accepts them, correctly."""
    var strict = ParserConfig.interoperable()
    _ = loads('{"a":1,"b":2}', strict)
    var raised = False
    try:
        _ = loads('{"a":1,"a":2}', strict)
    except:
        raised = True
    assert_true(raised)
    # Nested objects are checked too.
    raised = False
    try:
        _ = loads('{"x":[{"b":1,"b":2}]}', strict)
    except:
        raised = True
    assert_true(raised)
    # Without the mode this is a valid RFC 8259 document.
    assert_equal(Int(loads('{"a":1,"a":2}')["a"].int_value()), 1)


def test_ijson_rejects_unpaired_surrogates() raises:
    """RFC 7493 section 2.3.

    Checked against the source text, because parsing turns an unpaired
    escape into U+FFFD and that is indistinguishable from a literal one.
    """
    var strict = ParserConfig.interoperable()
    _ = loads('"\\ud834\\udd1e"', strict)
    _ = loads('"\\u0041"', strict)
    var bad_cases: List[String] = ['"\\ud800"', '"\\udc00"', '"\\ud800a"']
    for bad in bad_cases:
        var raised = False
        try:
            _ = loads(bad, strict)
        except:
            raised = True
        assert_true(raised, "should have been rejected: " + bad)


def test_ijson_keeps_exact_large_integers() raises:
    """RFC 7493 section 2.2 is advice to protocol designers.

    An integer outside the range a `Float64` names exactly is kept
    exact here, so rejecting it would lose information rather than
    protect anyone.
    """
    var strict = ParserConfig.interoperable()
    assert_equal(dumps(loads("9007199254740993", strict)), "9007199254740993")


def main() raises:
    print("=" * 60)
    print("test_api.mojo - Unified API Tests")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()

    # Cleanup. The files are written by the round-trip tests above;
    # removal is best-effort because a test that failed early may not
    # have created its file.
    try:
        remove("test_api.json")
        remove("test_api.ndjson")
        remove("test_api_stream.ndjson")
    except:
        pass
