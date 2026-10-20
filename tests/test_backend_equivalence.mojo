# Tests asserting CPU backend equivalence.
#
# Both the default Mojo native parser (`target="cpu"`) and the simdjson
# FFI parser (`target="cpu-simdjson"`) must produce structurally
# identical `Value` trees and identical `dumps()` output for the same
# input. This test guards against drift between the two
# implementations.

from std.testing import assert_true, TestSuite

from json import loads, dumps, Value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _values_equal(a: Value, b: Value) raises -> Bool:
    """Structural equality across two Values that may have been produced
    by different backends. Walks arrays and objects element-wise so
    that two backends emitting semantically equal but textually
    different raws do not false-mismatch."""
    if a.is_null():
        return b.is_null()
    if a.is_bool():
        return b.is_bool() and a.bool_value() == b.bool_value()
    if a.is_uint():
        return b.is_uint() and a.uint_value() == b.uint_value()
    if a.is_int():
        return b.is_int() and a.int_value() == b.int_value()
    if a.is_float():
        return b.is_float() and a.float_value() == b.float_value()
    if a.is_string():
        return b.is_string() and a.string_value() == b.string_value()
    if a.is_array():
        if not b.is_array():
            return False
        if a.array_count() != b.array_count():
            return False
        var n = a.array_count()
        for i in range(n):
            var ai = a[i]
            var bi = b[i]
            if not _values_equal(ai, bi):
                return False
        return True
    if a.is_object():
        if not b.is_object():
            return False
        var ak = a.object_keys()
        var bk = b.object_keys()
        if len(ak) != len(bk):
            return False
        for i in range(len(ak)):
            var key = ak[i]
            # Find the same key in b.
            var found = False
            for j in range(len(bk)):
                if bk[j] == key:
                    found = True
                    break
            if not found:
                return False
            var av = a[key]
            var bv = b[key]
            if not _values_equal(av, bv):
                return False
        return True
    return False


def _check_equivalence(s: String) raises:
    """Both backends must yield structurally equal Values and dumps that
    round-trip to structurally equal Values. The legacy CPU path and the
    simdjson FFI both echo input whitespace through `raw_json()`, but the
    tape-backed CPU path emits canonical compact JSON; comparing dumps
    byte-for-byte over-specifies behaviour. Re-parsing closes the gap."""
    var native = loads[target="cpu"](s)
    var ffi = loads[target="cpu-simdjson"](s)

    assert_true(
        _values_equal(native, ffi),
        "Backends produced structurally different Values for: " + s,
    )

    var dn = dumps(native)
    var df = dumps(ffi)
    var rn = loads[target="cpu"](dn)
    var rf = loads[target="cpu"](df)
    assert_true(
        _values_equal(rn, rf),
        (
            "Backends produced semantically different dumps for: "
            + s
            + " (native="
            + dn
            + " ffi="
            + df
            + ")"
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_primitives() raises:
    _check_equivalence("null")
    _check_equivalence("true")
    _check_equivalence("false")
    _check_equivalence("0")
    _check_equivalence("42")
    _check_equivalence("-7")


def test_simple_string() raises:
    _check_equivalence('"hello"')
    _check_equivalence('""')


def test_simple_array() raises:
    _check_equivalence("[]")
    _check_equivalence("[1,2,3]")
    _check_equivalence('["a","b"]')


def test_simple_object() raises:
    _check_equivalence("{}")
    _check_equivalence('{"a":1}')
    _check_equivalence('{"name":"Alice","age":30}')


def test_nested_object() raises:
    _check_equivalence(
        '{"user":{"name":"Alice","scores":[95,87,92]},"active":true}'
    )


def test_array_of_objects() raises:
    _check_equivalence(
        '[{"id":1,"name":"a"},{"id":2,"name":"b"},{"id":3,"name":"c"}]'
    )


def test_deeply_nested() raises:
    _check_equivalence('{"a":{"b":{"c":{"d":[1,[2,[3]]]}}}}')


def test_mixed_types() raises:
    _check_equivalence(
        '{"int":1,"float":3.14,"bool":true,"null":null,"str":"x"}'
    )


def test_floats_in_array() raises:
    _check_equivalence("[1.5, 2.25, 0.0]")


def test_unicode_string_basic() raises:
    """ASCII-only strings; non-ASCII normalization across the two
    backends is covered separately."""
    _check_equivalence('"hello world"')
    _check_equivalence('"with spaces and 123"')


def test_escapes_numbers_and_empty_containers() raises:
    _check_equivalence(
        '{"k\\"q":"a\\nb\\u00e9","big":1152921504606846976,'
        '"neg":-1152921504606846977,"u":18446744073709551615,'
        '"f":-2.5e-3,"e":[],"o":{},"n":null,"t":[true,false]}'
    )
    var v = loads[target="cpu-simdjson"](
        '{"k\\"q":"\\u00e9","x":[1,{"y":"z"}]}'
    )
    assert_true(v.get('k"q').value().string_value() == "\u00e9")
    assert_true(v.get("x").value()[1].get("y").value().string_value() == "z")


def test_invalid_input_raises() raises:
    for bad in ['{"a":', "[1,]", '{"a" 1}', ""]:
        var raised = False
        try:
            _ = loads[target="cpu-simdjson"](String(bad))
        except:
            raised = True
        assert_true(raised)


def main() raises:
    print("=" * 60)
    print("test_backend_equivalence.mojo")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
