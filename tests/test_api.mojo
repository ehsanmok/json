# Comprehensive tests for the unified API

from testing import assert_equal, assert_true, TestSuite

from src import (
    loads, dumps, load, dump,
    Value, ParserConfig, SerializerConfig,
    LazyValue, StreamingParser,
    apply_patch, merge_patch, jsonpath_query, validate, is_valid,
)


# =============================================================================
# loads() tests
# =============================================================================

def test_loads_basic():
    """Test basic JSON parsing."""
    var data = loads('{"name":"Alice","age":30}')
    assert_equal(data["name"].string_value(), "Alice")
    assert_equal(Int(data["age"].int_value()), 30)


def test_loads_with_config():
    """Test loads with ParserConfig."""
    var config = ParserConfig(allow_comments=True, allow_trailing_comma=True)
    var data = loads('{"a": 1,} // comment', config)
    assert_equal(Int(data["a"].int_value()), 1)


def test_loads_ndjson():
    """Test loads with format=ndjson."""
    var values = loads[format="ndjson"]('{"a":1}\n{"a":2}\n{"a":3}')
    assert_equal(len(values), 3)
    assert_equal(Int(values[0]["a"].int_value()), 1)
    assert_equal(Int(values[2]["a"].int_value()), 3)


def test_loads_lazy():
    """Test loads with lazy=True."""
    var lazy = loads[lazy=True]('{"users":[{"name":"Alice"},{"name":"Bob"}]}')
    var name = lazy.get("/users/0/name")
    assert_equal(name.string_value(), "Alice")


def test_loads_all_types():
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

def test_dumps_basic():
    """Test basic serialization."""
    var data = loads('{"a":1}')
    var s = dumps(data)
    assert_true(s.find('"a"') >= 0)
    assert_true(s.find("1") >= 0)


def test_dumps_pretty():
    """Test dumps with indentation."""
    var data = loads('{"a":1,"b":2}')
    var s = dumps(data, indent="  ")
    assert_true(s.find("\n") >= 0)


def test_dumps_config():
    """Test dumps with SerializerConfig."""
    var data = loads('{"url":"http://x.com"}')
    var config = SerializerConfig(escape_forward_slash=True)
    var s = dumps(data, config)
    assert_true(s.find("\\/") >= 0)


def test_dumps_ndjson():
    """Test dumps with format=ndjson."""
    var values = List[Value]()
    values.append(loads('{"a":1}'))
    values.append(loads('{"a":2}'))
    var s = dumps[format="ndjson"](values)
    assert_true(s.find("\n") >= 0)


def test_dumps_roundtrip():
    """Test serialization roundtrip."""
    var original = '{"name":"Alice","scores":[95,87,92]}'
    var data = loads(original)
    var serialized = dumps(data)
    var reparsed = loads(serialized)
    assert_equal(reparsed["name"].string_value(), "Alice")


# =============================================================================
# load()/dump() file tests
# =============================================================================

def test_load_dump_roundtrip():
    """Test file load/dump roundtrip."""
    var data = loads('{"test":123,"arr":[1,2,3]}')
    
    var f_out = open("test_api.json", "w")
    dump(data, f_out)
    f_out.close()
    
    var loaded = load("test_api.json")
    assert_equal(Int(loaded["test"].int_value()), 123)


def test_load_ndjson():
    """Test load auto-detects .ndjson files."""
    var f_out = open("test_api.ndjson", "w")
    f_out.write('{"a":1}\n{"a":2}\n')
    f_out.close()
    
    var data = load("test_api.ndjson")  # Auto-detects from extension
    assert_true(data.is_array())
    assert_equal(data.array_count(), 2)


def test_load_ndjson_gpu():
    """Test load with GPU + .ndjson auto-detection."""
    var f_out = open("test_api_gpu.ndjson", "w")
    f_out.write('{"x":1}\n{"x":2}\n{"x":3}\n')
    f_out.close()
    
    var data = load[target="gpu"]("test_api_gpu.ndjson")
    assert_true(data.is_array())
    assert_equal(data.array_count(), 3)


def test_loads_ndjson_gpu():
    """Test loads[format='ndjson'] with GPU."""
    var ndjson = '{"id":1}\n{"id":2}'
    var values = loads[target="gpu", format="ndjson"](ndjson)
    assert_equal(len(values), 2)


def test_load_streaming():
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

def test_value_access():
    """Test value access methods."""
    var data = loads('{"users":[{"name":"Alice"},{"name":"Bob"}]}')
    assert_equal(data["users"][0]["name"].string_value(), "Alice")
    assert_equal(data.at("/users/1/name").string_value(), "Bob")


def test_value_mutation():
    """Test value mutation."""
    var data = loads('{"a":1}')
    data.set("b", Value(2))
    assert_equal(Int(data["b"].int_value()), 2)


def test_value_iteration():
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

def test_jsonpath():
    """Test JSONPath queries."""
    var data = loads('{"users":[{"name":"Alice"},{"name":"Bob"}]}')
    var names = jsonpath_query(data, "$.users[*].name")
    assert_equal(len(names), 2)


def test_json_patch():
    """Test JSON Patch."""
    var doc = loads('{"a":1}')
    var patch = loads('[{"op":"add","path":"/b","value":2}]')
    var result = apply_patch(doc, patch)
    assert_equal(Int(result["b"].int_value()), 2)


def test_merge_patch():
    """Test JSON Merge Patch."""
    var target = loads('{"a":1,"b":2}')
    var patch = loads('{"b":null,"c":3}')
    var result = merge_patch(target, patch)
    assert_equal(Int(result["c"].int_value()), 3)


def test_schema_validation():
    """Test JSON Schema validation."""
    var schema = loads('{"type":"object","required":["name"]}')
    var valid_doc = loads('{"name":"Alice"}')
    var invalid_doc = loads('{"age":30}')
    
    assert_true(is_valid(valid_doc, schema))
    assert_true(not is_valid(invalid_doc, schema))


# =============================================================================
# NDJSON roundtrip
# =============================================================================

def test_ndjson_roundtrip():
    """Test NDJSON roundtrip."""
    var original = '{"id":1}\n{"id":2}\n{"id":3}'
    var values = loads[format="ndjson"](original)
    var serialized = dumps[format="ndjson"](values)
    var reparsed = loads[format="ndjson"](serialized)
    assert_equal(len(reparsed), 3)


def main():
    print("=" * 60)
    print("test_api.mojo - Unified API Tests")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
    
    # Cleanup
    import os
    try:
        os.remove("test_api.json")
        os.remove("test_api.ndjson")
        os.remove("test_api_gpu.ndjson")
        os.remove("test_api_stream.ndjson")
    except:
        pass
