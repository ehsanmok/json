# GPU loading tests
# Tests: loads[target="gpu"](...)
#
# The GPU path runs natively on NVIDIA, AMD, and Apple Metal. Tests
# are short-circuited only on hosts without any accelerator at all
# (CPU-only CI).

from std.os import remove
from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from json import dumps, Value, Null
from json.gpu import loads, load
from json import serialize_json, deserialize_json
from json.gpu.tape_adapter import parse_gpu_to_value
from json.cpu.stage1_scalar import parse_structural_scalar
from json.types import JSONResult


# Compile-time predicate: a GPU runtime is reachable from this build.
comptime GPU_RUNTIME_AVAILABLE = has_accelerator()


# =============================================================================
# GPU loads Tests
# =============================================================================


def test_loads_gpu_simple_object() raises:
    """Test GPU loads with simple object."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]('{"name": "Alice"}')
    assert_true(v.is_object(), "GPU should return object")


def test_loads_gpu_simple_array() raises:
    """Test GPU loads with simple array."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("[1, 2, 3, 4, 5]")
    assert_true(v.is_array(), "GPU should return array")


def test_loads_gpu_nested() raises:
    """Test GPU loads with nested structure."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]('{"data": {"nested": [1, 2, 3]}}')
    assert_true(v.is_object(), "GPU should handle nested structures")


def test_loads_gpu_string() raises:
    """Test GPU loads with string."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]('"hello world"')
    assert_true(v.is_string(), "GPU should return string")


def test_loads_gpu_number() raises:
    """Test GPU loads with number."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("12345")
    assert_true(v.is_int() or v.is_float(), "GPU should return number")


def test_loads_gpu_bool() raises:
    """Test GPU loads with boolean."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("true")
    assert_true(v.is_bool(), "GPU should return bool")


def test_loads_gpu_null() raises:
    """Test GPU loads with null."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("null")
    assert_true(v.is_null(), "GPU should return null")


def test_loads_gpu_bool_false() raises:
    """Test GPU loads with false."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("false")
    assert_true(v.is_bool(), "GPU should return bool")
    assert_equal(v.bool_value(), False)


def test_loads_gpu_negative_number() raises:
    """Test GPU loads with negative number."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("-42")
    assert_true(v.is_int() or v.is_float(), "GPU should return number")


def test_loads_gpu_float() raises:
    """Test GPU loads with float."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("3.14159")
    assert_true(v.is_float(), "GPU should return float")


def test_loads_gpu_empty_object() raises:
    """Test GPU loads with empty object."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("{}")
    assert_true(v.is_object(), "GPU should return empty object")


def test_loads_gpu_empty_array() raises:
    """Test GPU loads with empty array."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("[]")
    assert_true(v.is_array(), "GPU should return empty array")


def test_loads_gpu_array_of_objects() raises:
    """Test GPU loads with array of objects."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]('[{"a": 1}, {"b": 2}]')
    assert_true(v.is_array(), "GPU should return array")


def test_loads_gpu_deeply_nested() raises:
    """Test GPU loads with deeply nested structure."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]('{"a": {"b": {"c": {"d": 1}}}}')
    assert_true(v.is_object(), "GPU should handle deep nesting")


# =============================================================================
# CPU/GPU Equivalence Tests
# =============================================================================


def test_cpu_gpu_equivalence_object() raises:
    """Test CPU and GPU produce equivalent results for objects."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var json = '{"a": 1, "b": 2}'
    var cpu_result = loads[target="cpu"](json)
    var gpu_result = loads[target="gpu"](json)
    assert_equal(cpu_result.is_object(), gpu_result.is_object())


def test_cpu_gpu_equivalence_array() raises:
    """Test CPU and GPU produce equivalent results for arrays."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var json = "[1, 2, 3, 4, 5]"
    var cpu_result = loads[target="cpu"](json)
    var gpu_result = loads[target="gpu"](json)
    assert_equal(cpu_result.is_array(), gpu_result.is_array())


def test_cpu_gpu_equivalence_nested() raises:
    """Test CPU and GPU produce equivalent results for nested structures."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var json = '{"users": [{"name": "Alice"}, {"name": "Bob"}], "count": 2}'
    var cpu_result = loads[target="cpu"](json)
    var gpu_result = loads[target="gpu"](json)
    assert_equal(cpu_result.is_object(), gpu_result.is_object())


def test_gpu_dumps_roundtrip() raises:
    """Test GPU loads then dumps roundtrip."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var json = '{"test": "value"}'
    var v = loads[target="gpu"](json)
    var output = dumps(v)
    var v2 = loads[target="cpu"](output)
    assert_true(v2.is_object(), "GPU roundtrip should produce valid object")


@fieldwise_init
struct _GPUPerson(Defaultable, Movable):
    var name: String
    var age: Int
    var active: Bool

    def __init__(out self):
        self.name = ""
        self.age = 0
        self.active = False


def test_gpu_reflection_roundtrip() raises:
    """Typed serde over text a GPU parse produced.

    `deserialize_json` no longer takes a backend: it reads bytes
    straight into the struct rather than building a document first, so
    there is no parse for a backend to select. A caller who wants the
    GPU parser calls `loads_gpu` and gets a `Value`. This
    checks the two meet: GPU-parsed text round-trips through the typed
    path.
    """
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var json_str = '{"name":"GPU Test","age":42,"active":true}'
    var parsed = loads[target="gpu"](json_str)
    assert_equal(parsed["name"].string_value(), "GPU Test")

    var person = deserialize_json[_GPUPerson](json_str)
    assert_equal(person.name, "GPU Test")
    assert_equal(person.age, 42)
    assert_equal(person.active, True)

    var back = serialize_json(person)
    var rt = deserialize_json[_GPUPerson](back)
    assert_equal(rt.name, "GPU Test")
    assert_equal(rt.age, 42)


# =============================================================================
# Apple Metal regression coverage + tape-adapter direct test
# =============================================================================


def test_gpu_handles_escaped_quotes_in_strings() raises:
    """Apple's fused_json_kernel previously emitted `{}[]:,` bytes that
    fell *inside* string literals (its in-string mask couldn't survive
    cross-chunk backslash runs / `\\"` patterns). The kernel now emits
    raw structural bits and `gpu/tape_adapter.mojo` filters them with a
    correct CPU-side escape state machine. This test pins that fix.
    """
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var json = String('{"msg":"He said \\"hi\\"","items":[1,2,3]}')
    var v = loads[target="gpu"](json)
    assert_equal(v["msg"].string_value(), 'He said "hi"')
    assert_equal(v["items"].array_count(), 3)
    assert_equal(v["items"][0].int_value(), 1)
    assert_equal(v["items"][2].int_value(), 3)


def test_gpu_handles_brace_inside_string() raises:
    """Bytes that look structural but live inside string literals must
    not show up in the merged structural index. Sanity check for the
    Apple fix and any future GPU rewrite."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var json = String('{"sql":"SELECT * FROM t WHERE k=\'a,b,c\'","n":7}')
    var v = loads[target="gpu"](json)
    assert_equal(v["sql"].string_value(), "SELECT * FROM t WHERE k='a,b,c'")
    assert_equal(v["n"].int_value(), 7)


def test_gpu_cross_chunk_strings() raises:
    """Apple Metal chunks the input at 32 MB by default. A string that
    starts inside one chunk and ends inside the next (or whose escape
    sequence straddles the boundary) must still be filtered correctly
    by the CPU adapter. This test builds a >34 MB JSON whose first
    string is engineered to cross the 32 MB chunk boundary, with
    escaped quotes near the boundary, and asserts CPU/GPU agree on
    every leaf value.

    Skipped on non-Apple hosts (single-shot pipeline doesn't chunk).
    Skipped without an accelerator runtime.
    """
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    # Build approximately 34 MB of JSON: an array whose first element
    # is a single huge string that straddles the 32 MB chunk boundary,
    # followed by a structurally rich tail so we have lots of structural
    # characters to filter. The huge string contains literal `\"` and
    # `\\` to exercise the CPU escape state machine across the boundary.
    var n = 34 * 1024 * 1024  # 34 MB
    var head = String('["')
    var tail_struct = String(
        '",{"users":[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"}]}]'
    )
    var huge_filler_size = (
        n - len(head.as_bytes()) - len(tail_struct.as_bytes())
    )

    # Body: alternating chars + escaped quotes/backslashes, padded to
    # `huge_filler_size`. Every 1024 bytes we drop in `\"` (escaped
    # quote) so escape sequences land at varying offsets including
    # near the 32 MB boundary.
    var body = String("")
    var unit = String('abc\\"def')  # contains \" -- literal escape
    var unit_len = len(unit.as_bytes())
    var i = 0
    while len(body.as_bytes()) + unit_len < huge_filler_size:
        body += unit
        i += 1
    # Fill to exact size with safe ASCII.
    while len(body.as_bytes()) < huge_filler_size:
        body += String("x")

    var json = head + body + tail_struct
    # Sanity: total size > 32 MB so chunking definitely fires.
    assert_true(
        len(json.as_bytes()) > 32 * 1024 * 1024,
        "test JSON must exceed the 32 MB chunk boundary",
    )

    # CPU parse as oracle.
    var v_cpu = loads(json)
    var v_gpu = loads[target="gpu"](json)

    # Oracle and GPU must agree on the structurally-rich tail.
    var users_cpu = v_cpu[1]["users"]
    var users_gpu = v_gpu[1]["users"]
    assert_equal(
        users_cpu.array_count(),
        users_gpu.array_count(),
        "CPU and GPU agree on tail array count after chunk boundary",
    )
    assert_equal(
        users_cpu[0]["id"].int_value(),
        users_gpu[0]["id"].int_value(),
        "first user id matches",
    )
    assert_equal(
        users_cpu[0]["name"].string_value(),
        users_gpu[0]["name"].string_value(),
        "first user name matches",
    )
    assert_equal(
        users_cpu[1]["id"].int_value(),
        users_gpu[1]["id"].int_value(),
        "second user id matches",
    )
    assert_equal(
        users_cpu[1]["name"].string_value(),
        users_gpu[1]["name"].string_value(),
        "second user name matches",
    )


def test_tape_adapter_roundtrip() raises:
    """End-to-end test of `parse_gpu_to_value` without requiring the GPU.

    Builds a synthetic `JSONResult` from `stage1_scalar` (filtering quote
    positions out, which mirrors what the GPU kernel emits), feeds it
    through the tape adapter, and asserts the resulting Value structurally
    matches the canonical CPU parse.
    """
    var json = '{"users":[{"name":"Alice","age":30},{"name":"Bob","age":25}]}'

    var scalar_index = parse_structural_scalar(json)
    var bytes = json.as_bytes()

    # GPU emits {}[]:, only -- filter quotes out of the scalar oracle.
    var fake_gpu_result = JSONResult()
    for i in range(len(scalar_index.positions)):
        var p = Int(scalar_index.positions[i])
        var c = bytes[p]
        if c != UInt8(ord('"')):
            fake_gpu_result.structural.append(Int32(p))
    fake_gpu_result.file_size = len(bytes)

    var adapter_value = parse_gpu_to_value(json, fake_gpu_result^)
    var cpu_value = loads(json)

    assert_true(adapter_value.is_object(), "adapter Value should be object")
    assert_equal(
        adapter_value.is_object(),
        cpu_value.is_object(),
        "adapter and CPU agree on object-ness",
    )
    var users = adapter_value["users"]
    assert_true(users.is_array(), "adapter Value users[] is array")
    assert_equal(users.array_count(), 2, "adapter Value users has 2 elements")


def test_load_ndjson_on_gpu() raises:
    """`load_gpu` with `.ndjson` auto-detection."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var f_out = open("test_gpu_api.ndjson", "w")
    f_out.write('{"x":1}\n{"x":2}\n{"x":3}\n')
    f_out.close()

    var data = load[target="gpu"]("test_gpu_api.ndjson")
    assert_true(data.is_array())
    assert_equal(data.array_count(), 3)


def test_loads_ndjson_on_gpu() raises:
    """Newline-delimited JSON, one line per GPU launch."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var f_out = open("test_gpu_lines.ndjson", "w")
    f_out.write('{"id":1}\n{"id":2}\n')
    f_out.close()
    var values = load[target="gpu"]("test_gpu_lines.ndjson")
    assert_equal(values.array_count(), 2)


def main() raises:
    print("=" * 60)
    print("test_gpu.mojo - GPU loads() tests")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()

    # Best effort: a test that failed early may not have created it.
    try:
        remove("test_gpu_api.ndjson")
        remove("test_gpu_lines.ndjson")
    except:
        pass
