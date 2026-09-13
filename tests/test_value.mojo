# Tests for json/value.mojo

from std.collections import Dict, List
from std.testing import assert_equal, assert_true, assert_false, TestSuite

from json import Value, Null, loads


def test_null_creation() raises:
    """Test null value creation."""
    var v = Value(None)
    assert_true(v.is_null(), "Should be null")
    assert_equal(String(v), "null")


def test_null_from_null_type() raises:
    """Test null from Null type."""
    var v = Value(Null())
    assert_true(v.is_null(), "Should be null")


def test_bool_true() raises:
    """Test boolean true."""
    var v = Value(True)
    assert_true(v.is_bool(), "Should be bool")
    assert_true(v.bool_value(), "Should be true")
    assert_equal(String(v), "true")


def test_bool_false() raises:
    """Test boolean false."""
    var v = Value(False)
    assert_true(v.is_bool(), "Should be bool")
    assert_false(v.bool_value(), "Should be false")
    assert_equal(String(v), "false")


def test_int_positive() raises:
    """Test positive integer."""
    var v = Value(42)
    assert_true(v.is_int(), "Should be int")
    assert_true(v.is_number(), "Should be number")
    assert_equal(Int(v.int_value()), 42)


def test_int_negative() raises:
    """Test negative integer."""
    var v = Value(-123)
    assert_true(v.is_int(), "Should be int")
    assert_equal(Int(v.int_value()), -123)


def test_int_zero() raises:
    """Test zero."""
    var v = Value(0)
    assert_true(v.is_int(), "Should be int")
    assert_equal(Int(v.int_value()), 0)


def test_float() raises:
    """Test float value."""
    var v = Value(3.14)
    assert_true(v.is_float(), "Should be float")
    assert_true(v.is_number(), "Should be number")


def test_string() raises:
    """Test string value."""
    var v = Value("hello")
    assert_true(v.is_string(), "Should be string")
    assert_equal(v.string_value(), "hello")


def test_string_empty() raises:
    """Test empty string."""
    var v = Value("")
    assert_true(v.is_string(), "Should be string")
    assert_equal(v.string_value(), "")


def test_equality_null() raises:
    """Test null equality."""
    var a = Value(None)
    var b = Value(None)
    assert_true(a == b, "Nulls should be equal")


def test_equality_bool() raises:
    """Test bool equality."""
    var a = Value(True)
    var b = Value(True)
    assert_true(a == b, "Bools should be equal")


def test_equality_int() raises:
    """Test int equality."""
    var a = Value(42)
    var b = Value(42)
    assert_true(a == b, "Ints should be equal")


def test_equality_string() raises:
    """Test string equality."""
    var a = Value("hello")
    var b = Value("hello")
    assert_true(a == b, "Strings should be equal")


def test_inequality() raises:
    """Test inequality."""
    var a = Value(1)
    var b = Value(2)
    assert_true(a != b, "Different values should not be equal")


def test_type_mismatch() raises:
    """Test type mismatch."""
    var a = Value(1)
    var b = Value("1")
    assert_true(a != b, "Different types should not be equal")


# JSON Pointer (RFC 6901) tests
def test_json_pointer_empty() raises:
    """Test empty pointer returns whole document."""
    from json import loads

    var data = loads('{"a":1}')
    var result = data.at("")
    assert_true(
        result.is_object(), "Empty pointer should return whole document"
    )


def test_json_pointer_simple_object() raises:
    """Test simple object access."""
    from json import loads

    var data = loads('{"name":"Alice","age":30}')
    var name = data.at("/name")
    assert_true(name.is_string(), "Should be string")
    assert_equal(name.string_value(), "Alice")

    var age = data.at("/age")
    assert_true(age.is_int(), "Should be int")
    assert_equal(Int(age.int_value()), 30)


def test_json_pointer_nested_object() raises:
    """Test nested object access."""
    from json import loads

    var data = loads('{"user":{"name":"Bob","email":"bob@test.com"}}')
    var name = data.at("/user/name")
    assert_equal(name.string_value(), "Bob")

    var email = data.at("/user/email")
    assert_equal(email.string_value(), "bob@test.com")


def test_json_pointer_array_index() raises:
    """Test array index access."""
    from json import loads

    var data = loads('{"items":[10,20,30]}')
    var first = data.at("/items/0")
    assert_equal(Int(first.int_value()), 10)

    var second = data.at("/items/1")
    assert_equal(Int(second.int_value()), 20)

    var third = data.at("/items/2")
    assert_equal(Int(third.int_value()), 30)


def test_json_pointer_array_of_objects() raises:
    """Test array of objects access."""
    from json import loads

    var data = loads('{"users":[{"name":"Alice"},{"name":"Bob"}]}')
    var first_name = data.at("/users/0/name")
    assert_equal(first_name.string_value(), "Alice")

    var second_name = data.at("/users/1/name")
    assert_equal(second_name.string_value(), "Bob")


def test_json_pointer_escape_tilde() raises:
    """Test ~0 escape for tilde."""
    from json import loads

    var data = loads('{"a~b":42}')
    var result = data.at("/a~0b")
    assert_equal(Int(result.int_value()), 42)


def test_json_pointer_escape_slash() raises:
    """Test ~1 escape for slash."""
    from json import loads

    var data = loads('{"a/b":42}')
    var result = data.at("/a~1b")
    assert_equal(Int(result.int_value()), 42)


def test_json_pointer_deep_nesting() raises:
    """Test deeply nested access."""
    from json import loads

    var data = loads('{"a":{"b":{"c":{"d":"deep"}}}}')
    var result = data.at("/a/b/c/d")
    assert_equal(result.string_value(), "deep")


def test_json_pointer_null_value() raises:
    """Test accessing null value."""
    from json import loads

    var data = loads('{"value":null}')
    var result = data.at("/value")
    assert_true(result.is_null(), "Should be null")


def test_json_pointer_bool_value() raises:
    """Test accessing boolean value."""
    from json import loads

    var data = loads('{"active":true,"deleted":false}')
    var active = data.at("/active")
    assert_true(active.is_bool() and active.bool_value(), "Should be true")

    var deleted = data.at("/deleted")
    assert_true(
        deleted.is_bool() and not deleted.bool_value(), "Should be false"
    )


# Value iteration tests
def test_array_items() raises:
    """Test iterating over array items."""
    from json import loads

    var data = loads("[1, 2, 3]")
    var items = data.array_items()
    assert_equal(len(items), 3)
    assert_equal(Int(items[0].int_value()), 1)
    assert_equal(Int(items[1].int_value()), 2)
    assert_equal(Int(items[2].int_value()), 3)


def test_array_items_mixed() raises:
    """Test iterating over mixed array items."""
    from json import loads

    var data = loads('[1, "hello", true, null]')
    var items = data.array_items()
    assert_equal(len(items), 4)
    assert_true(items[0].is_int())
    assert_true(items[1].is_string())
    assert_true(items[2].is_bool())
    assert_true(items[3].is_null())


def test_object_items() raises:
    """Test iterating over object items."""
    from json import loads

    var data = loads('{"a": 1, "b": 2}')
    var items = data.object_items()
    assert_equal(len(items), 2)


def test_array_getitem() raises:
    """Test array index access."""
    from json import loads

    var data = loads("[10, 20, 30]")
    assert_equal(Int(data[0].int_value()), 10)
    assert_equal(Int(data[1].int_value()), 20)
    assert_equal(Int(data[2].int_value()), 30)


def test_object_getitem() raises:
    """Test object key access."""
    from json import loads

    var data = loads('{"name": "Alice", "age": 30}')
    assert_equal(data["name"].string_value(), "Alice")
    assert_equal(Int(data["age"].int_value()), 30)


def test_nested_access() raises:
    """Test nested array/object access."""
    from json import loads

    var data = loads('{"users": [{"name": "Alice"}, {"name": "Bob"}]}')
    var users = data["users"]
    assert_true(users.is_array())
    var first = users[0]
    assert_equal(first["name"].string_value(), "Alice")


# Value mutation tests
def test_object_set_new_key() raises:
    """Test adding a new key to an object."""
    from json import loads

    var data = loads('{"name": "Alice"}')
    data.set("age", Value(30))
    assert_equal(data.object_count(), 2)


def test_object_set_update_key() raises:
    """Test updating an existing key."""
    from json import loads

    var data = loads('{"name": "Alice"}')
    data.set("name", Value("Bob"))
    assert_equal(data["name"].string_value(), "Bob")


def test_array_set() raises:
    """Test setting array element."""
    from json import loads

    var data = loads("[1, 2, 3]")
    data.set(1, Value(20))
    assert_equal(Int(data[1].int_value()), 20)


def test_array_append() raises:
    """Test appending to array."""
    from json import loads

    var data = loads("[1, 2]")
    data.append(Value(3))
    assert_equal(data.array_count(), 3)


def test_array_append_empty() raises:
    """Test appending to empty array."""
    from json import loads

    var data = loads("[]")
    data.append(Value(1))
    assert_equal(data.array_count(), 1)
    assert_equal(Int(data[0].int_value()), 1)


def test_object_set_empty() raises:
    """Test adding to empty object."""
    from json import loads

    var data = loads("{}")
    data.set("key", Value("value"))
    assert_equal(data.object_count(), 1)


# ---------------------------------------------------------------------------
# Structural equality (bug 1).
#
# `__eq__` used to compare the SERIALIZED TEXT of an array or object,
# so two spellings of one JSON value came out unequal. These pin the
# two shapes that broke.
# ---------------------------------------------------------------------------


def test_eq_object_member_order_is_ignored() raises:
    """`{"a":1,"b":2}` equals `{"b":2,"a":1}`.

    The regression test for the text comparison: an object's members
    carry no order, and the old `__eq__` said these differed.
    """
    var a = loads('{"a":1,"b":2}')
    var b = loads('{"b":2,"a":1}')
    assert_true(a == b, "member order must not affect equality")
    assert_false(a != b, "__ne__ must agree with __eq__")


def test_eq_number_one_equals_one_point_zero() raises:
    """`[1]` equals `[1.0]`.

    The other half of the text comparison: JSON has one number type,
    and the old `__eq__` compared "1" against "1.0".
    """
    var a = loads("[1]")
    var b = loads("[1.0]")
    assert_true(a == b, "1 and 1.0 are the same JSON number")
    assert_true(Value(1) == Value(1.0), "scalars compare by value too")
    assert_true(Value(0) == Value(-0.0), "negative zero is zero")


def test_eq_number_rejects_fractional_and_wrong_value() raises:
    """A float equals an integer only when it is exactly that integer."""
    assert_true(Value(1) != Value(1.5), "1 is not 1.5")
    assert_true(Value(2) != Value(1.0), "2 is not 1.0")


def test_eq_uint_is_not_ignored() raises:
    """An integer above `Int64.MAX` compares in the unsigned domain.

    `is_uint` was invisible to the old comparison. A magnitude the
    signed range cannot name must equal itself, must equal the same
    magnitude written as a float, and must never equal a negative.
    """
    var big = Value(UInt64(18446744073709551615))
    var same = Value(UInt64(18446744073709551615))
    assert_true(big == same, "an unsigned magnitude equals itself")
    assert_true(big != Value(-1), "an unsigned magnitude is not -1")
    assert_true(Value(UInt64(7)) == Value(7), "7 is 7 in either range")
    assert_true(Value(UInt64(7)) == Value(7.0), "and as a float")


def test_eq_nested_order_insensitivity() raises:
    """Member order is ignored at every depth, not only at the root."""
    var a = loads('{"x":{"p":1,"q":[1,{"m":1,"n":2}]}}')
    var b = loads('{"x":{"q":[1,{"n":2,"m":1}],"p":1}}')
    assert_true(a == b, "nested objects compare structurally")


def test_eq_array_order_still_matters() raises:
    """An array is ordered, so `[1,2]` differs from `[2,1]`."""
    assert_true(loads("[1,2]") != loads("[2,1]"), "arrays keep their order")


def test_eq_across_representations() raises:
    """A parsed value equals an equal hand-built one, and the reverse."""
    var parsed = loads('{"a":[1,2],"b":"z"}')
    var built = Value.object()
    var arr = Value.array()
    arr.append(Value(1))
    arr.append(Value(2))
    built.set("b", Value("z"))
    built.set("a", arr)
    assert_true(parsed == built, "view equals owned")
    assert_true(built == parsed, "and owned equals view")


def test_eq_rejects_different_types_and_sizes() raises:
    """Type and cardinality mismatches are unequal."""
    assert_true(loads('{"a":1}') != loads("[1]"), "object is not array")
    assert_true(loads("[1,2]") != loads("[1]"), "lengths differ")
    assert_true(loads('{"a":1}') != loads('{"b":1}'), "names differ")
    assert_true(loads("null") != loads("false"), "null is not false")
    assert_true(loads('"1"') != loads("1"), "a string is not a number")


# ---------------------------------------------------------------------------
# Hashing.
# ---------------------------------------------------------------------------


def test_hash_agrees_with_equality() raises:
    """Equal values hash equal, including reordered objects."""
    assert_equal(
        loads('{"a":1,"b":2}').hash_u64(), loads('{"b":2,"a":1}').hash_u64()
    )
    assert_equal(Value(1).hash_u64(), Value(1.0).hash_u64())
    assert_equal(Value(UInt64(7)).hash_u64(), Value(7).hash_u64())
    var built = Value.object()
    built.set("a", Value(1))
    assert_equal(loads('{"a":1}').hash_u64(), built.hash_u64())


def test_value_keys_a_dict() raises:
    """A `Value` can key a `Dict`, and lookup ignores member order."""
    var table = Dict[Value, String]()
    table[loads('{"a":1,"b":2}')] = "first"
    table[Value("k")] = "second"
    assert_equal(table[loads('{"b":2,"a":1}')], "first")
    assert_equal(table[Value("k")], "second")
    assert_equal(len(table), 2)


# ---------------------------------------------------------------------------
# len, bool, contains.
# ---------------------------------------------------------------------------


def test_len_of_containers_and_string() raises:
    """`len` counts elements, members, and a string's bytes."""
    assert_equal(len(loads("[1,2,3]")), 3)
    assert_equal(len(loads('{"a":1,"b":2}')), 2)
    assert_equal(len(loads("[]")), 0)
    assert_equal(len(Value("hello")), 5)
    assert_equal(len(Value("")), 0)


def test_len_of_scalar_raises() raises:
    """A null, boolean or number has no length and says so."""
    var raised = False
    try:
        _ = len(Value(1))
    except e:
        raised = True
        assert_true(
            String(e).find("integer") >= 0, "the error names the type found"
        )
    assert_true(raised, "len of a number must raise")

    raised = False
    try:
        _ = len(Value(None))
    except:
        raised = True
    assert_true(raised, "len of null must raise")


def test_bool_truthiness() raises:
    """Truthiness follows the rule every JSON-speaking language uses."""
    assert_false(Bool(Value(None)), "null is falsy")
    assert_false(Bool(Value(False)), "false is falsy")
    assert_true(Bool(Value(True)), "true is truthy")
    assert_false(Bool(Value(0)), "zero is falsy")
    assert_true(Bool(Value(1)), "a non-zero number is truthy")
    assert_false(Bool(Value(0.0)), "zero as a float is falsy")
    assert_false(Bool(Value("")), "the empty string is falsy")
    assert_true(Bool(Value("x")), "a non-empty string is truthy")
    assert_false(Bool(loads("[]")), "an empty array is falsy")
    assert_true(Bool(loads("[0]")), "a non-empty array is truthy")
    assert_false(Bool(loads("{}")), "an empty object is falsy")
    assert_true(Bool(loads('{"a":null}')), "a non-empty object is truthy")


def test_contains_object_key_and_array_element() raises:
    """`"key" in obj` and `3 in arr` both work."""
    var obj = loads('{"name":"Alice","age":30}')
    assert_true("name" in obj, "an existing member is found")
    assert_false("nope" in obj, "an absent member is not")

    var arr = loads("[1, 2, 3]")
    assert_true(3 in arr, "an integer element is found")
    assert_false(9 in arr, "a missing one is not")
    assert_true(Value(3.0) in arr, "and equality is structural")

    var strings = loads('["a","b"]')
    assert_true("a" in strings, "a string element is found in an array")
    assert_false("c" in strings, "a missing one is not")


def test_contains_structural_element() raises:
    """Membership of a container element ignores member order."""
    var arr = loads('[{"a":1,"b":2}]')
    assert_true(loads('{"b":2,"a":1}') in arr, "reordered members still match")
    assert_false(loads('{"a":1}') in arr, "a different object does not")


def test_contains_on_a_scalar_is_false() raises:
    """A scalar contains nothing and answers false rather than raising."""
    assert_false("a" in Value(1), "a number has no members")
    assert_false(1 in Value("a"), "nor does a string hold elements")


# ---------------------------------------------------------------------------
# Iteration.
# ---------------------------------------------------------------------------


def test_iterate_array_lazily() raises:
    """`for item in arr:` walks the elements in order."""
    var arr = loads('[1, "two", true]')
    var seen = 0
    for item in arr:
        if seen == 0:
            assert_equal(item.int_value(), 1)
        elif seen == 1:
            assert_equal(item.string_value(), "two")
        else:
            assert_true(item.bool_value(), "the third element is true")
        seen += 1
    assert_equal(seen, 3)


def test_iterate_owned_array() raises:
    """Iteration works on a hand-built array too."""
    var arr = Value.array()
    arr.append(Value(10))
    arr.append(Value(20))
    var total = Int64(0)
    for item in arr:
        total += item.int_value()
    assert_equal(total, 30)


def test_iterate_non_array_raises() raises:
    """Iterating an object is spelled `items()`, so `__iter__` refuses."""
    var obj = loads('{"a":1}')
    var raised = False
    try:
        for _ in obj:
            pass
    except e:
        raised = True
        assert_true(String(e).find("object") >= 0, "the error names the type")
    assert_true(raised, "iterating an object must raise")


def test_object_items_keys_values() raises:
    """`items()`, `keys()` and `values()` walk an object's members."""
    var obj = loads('{"a":1,"b":2}')

    var names = List[String]()
    for key in obj.keys():
        names.append(key)
    assert_equal(len(names), 2)
    assert_equal(names[0], "a")
    assert_equal(names[1], "b")

    var total = Int64(0)
    for value in obj.values():
        total += value.int_value()
    assert_equal(total, 3)

    var pairs = 0
    for pair in obj.items():
        if pairs == 0:
            assert_equal(pair[0], "a")
            assert_equal(pair[1].int_value(), 1)
        else:
            assert_equal(pair[0], "b")
            assert_equal(pair[1].int_value(), 2)
        pairs += 1
    assert_equal(pairs, 2)


def test_object_iterators_on_owned_tree() raises:
    """The object iterators read a hand-built tree as well."""
    var obj = Value.object()
    obj.set("x", Value("one"))
    obj.set("y", Value("two"))
    var joined = String()
    for pair in obj.items():
        joined += pair[0]
        joined += "="
        joined += pair[1].string_value()
        joined += ";"
    assert_equal(joined, "x=one;y=two;")


def test_items_on_non_object_raises() raises:
    """`items()` / `keys()` / `values()` refuse a non-object."""
    var arr = loads("[1]")
    var raised = 0
    try:
        _ = arr.items()
    except:
        raised += 1
    try:
        _ = arr.keys()
    except:
        raised += 1
    try:
        _ = arr.values()
    except:
        raised += 1
    assert_equal(raised, 3)


# ---------------------------------------------------------------------------
# get / raw_member.
# ---------------------------------------------------------------------------


def test_get_returns_optional() raises:
    """`get` answers `None` for an absent key instead of raising."""
    var obj = loads('{"name":"Alice"}')
    var hit = obj.get("name")
    assert_true(Bool(hit), "an existing member is found")
    assert_equal(hit.value().string_value(), "Alice")

    var miss = obj.get("age")
    assert_false(Bool(miss), "an absent member is None, not an error")


def test_get_with_default() raises:
    """`get(key, default)` substitutes the fallback for an absent key."""
    var obj = loads('{"port":9000}')
    assert_equal(obj.get("port", Value(8080)).int_value(), 9000)
    assert_equal(
        obj.get("host", Value("localhost")).string_value(), "localhost"
    )


def test_get_on_non_object_raises() raises:
    """A member lookup on an array is a bug, not a miss."""
    var raised = False
    try:
        _ = loads("[1]").get("a")
    except:
        raised = True
    assert_true(raised, "get on an array must raise")


def test_raw_member_preserves_old_behaviour() raises:
    """`raw_member` is the pre-0.4.0 `get`: raw JSON text, raising."""
    var obj = loads('{"a":{"b":1},"c":2}')
    assert_equal(obj.raw_member("a"), '{"b":1}')
    assert_equal(obj.raw_member("c"), "2")
    var raised = False
    try:
        _ = obj.raw_member("zz")
    except:
        raised = True
    assert_true(raised, "raw_member still raises on an absent key")


# ---------------------------------------------------------------------------
# Typed accessors.
# ---------------------------------------------------------------------------


def test_as_accessors_raise_and_name_the_type() raises:
    """`as_*` checks the tag, unlike the silent `*_value` readers."""
    var text = Value("nope")
    assert_equal(text.int_value(), 0)  # The documented silent reading.

    var raised = False
    try:
        _ = text.as_int()
    except e:
        raised = True
        assert_true(String(e).find("string") >= 0, "the error names 'string'")
    assert_true(raised, "as_int on a string must raise")

    assert_equal(Value(42).as_int(), 42)
    assert_equal(Value(True).as_bool(), True)
    assert_equal(Value("hi").as_string(), "hi")
    assert_equal(Value(1.5).as_float(), 1.5)
    assert_equal(Value(3).as_float(), 3.0)
    assert_equal(Value(UInt64(9)).as_uint(), UInt64(9))
    assert_equal(Value(9).as_uint(), UInt64(9))


def test_as_uint_rejects_a_negative() raises:
    """A negative integer is refused rather than wrapped."""
    var raised = False
    try:
        _ = Value(-1).as_uint()
    except:
        raised = True
    assert_true(raised, "as_uint on -1 must raise")


def test_as_string_does_not_stringify() raises:
    """`as_string` refuses a number rather than formatting it."""
    var raised = False
    try:
        _ = Value(42).as_string()
    except:
        raised = True
    assert_true(raised, "as_string on a number must raise")


def test_or_accessors_fall_back() raises:
    """The `*_or` family substitutes a default instead of raising."""
    var text = Value("nope")
    assert_equal(text.int_or(-1), -1)
    assert_equal(text.bool_or(True), True)
    assert_equal(text.float_or(2.5), 2.5)
    assert_equal(text.uint_or(UInt64(4)), UInt64(4))
    assert_equal(Value(7).string_or("fallback"), "fallback")
    assert_equal(Value(7).int_or(-1), 7)
    assert_equal(Value(-1).uint_or(UInt64(4)), UInt64(4))
    assert_equal(Value(7).float_or(0.0), 7.0)


# ---------------------------------------------------------------------------
# Construction.
# ---------------------------------------------------------------------------


def test_construct_uint() raises:
    """`Value(UInt64)` reaches the range only `is_uint` could read."""
    var v = Value(UInt64(18446744073709551615))
    assert_true(v.is_uint(), "the unsigned tag is used")
    assert_equal(v.uint_value(), UInt64(18446744073709551615))
    assert_equal(String(v), "18446744073709551615")


def test_construct_sized_integers_and_float32() raises:
    """The narrow scalar types widen into a JSON number."""
    assert_equal(Value(Int8(-8)).int_value(), -8)
    assert_equal(Value(Int16(-16)).int_value(), -16)
    assert_equal(Value(Int32(-32)).int_value(), -32)
    assert_equal(Value(UInt8(8)).int_value(), 8)
    assert_equal(Value(UInt16(16)).int_value(), 16)
    assert_equal(Value(UInt32(32)).int_value(), 32)
    assert_true(Value(Float32(0.5)).is_float(), "Float32 becomes a number")
    assert_equal(Value(Float32(0.5)).float_value(), 0.5)


def test_construct_from_list_and_dict() raises:
    """Building `{"a":[1,2,3]}` takes two statements, not six."""
    var members = Dict[String, Value]()
    members["a"] = Value([Value(1), Value(2), Value(3)])
    var doc = Value(members^)
    assert_true(doc.is_object(), "a Dict becomes an object")
    assert_equal(doc.object_count(), 1)
    assert_true(doc == loads('{"a":[1,2,3]}'), "and holds the right JSON")

    var arr = Value([Value("x"), Value(None), Value(True)])
    assert_equal(arr.array_count(), 3)
    assert_equal(String(arr), '["x",null,true]')


# ---------------------------------------------------------------------------
# Indexing and pointers.
# ---------------------------------------------------------------------------


def test_negative_index() raises:
    """`arr[-1]` is the last element, as it is for `List`."""
    var arr = loads("[10, 20, 30]")
    assert_equal(arr[-1].int_value(), 30)
    assert_equal(arr[-3].int_value(), 10)
    var raised = False
    try:
        _ = arr[-4]
    except e:
        raised = True
        assert_true(String(e).find("-4") >= 0, "the error names what was asked")
    assert_true(raised, "an index past the start still raises")


def test_negative_index_on_owned_array() raises:
    """Negative indexing reads a hand-built array too."""
    var arr = Value.array()
    arr.append(Value(1))
    arr.append(Value(2))
    assert_equal(arr[-1].int_value(), 2)


def test_try_at_returns_none_for_a_miss() raises:
    """`try_at` is the non-raising twin of `at`."""
    var data = loads('{"a":{"b":[1,2]}}')
    assert_equal(data.try_at("/a/b/1").value().int_value(), 2)
    assert_true(data.try_at("").value().is_object(), "'' names the root")
    assert_false(Bool(data.try_at("/a/zz")), "a missing member is None")
    assert_false(Bool(data.try_at("/a/b/9")), "an out-of-range index is None")
    assert_false(Bool(data.try_at("/a/b/1/deeper")), "so is a scalar parent")


def test_try_at_still_rejects_a_malformed_pointer() raises:
    """A pointer that is not a pointer is a bug and still raises."""
    var raised = False
    try:
        _ = loads('{"a":1}').try_at("a")
    except:
        raised = True
    assert_true(raised, "a pointer missing its leading solidus must raise")


def test_type_name() raises:
    """`type_name` spells the JSON type the way JSON does."""
    assert_equal(Value(None).type_name(), "null")
    assert_equal(Value(True).type_name(), "boolean")
    assert_equal(Value(1).type_name(), "integer")
    assert_equal(Value(UInt64(1)).type_name(), "integer")
    assert_equal(Value(1.5).type_name(), "number")
    assert_equal(Value("s").type_name(), "string")
    assert_equal(loads("[]").type_name(), "array")
    assert_equal(loads("{}").type_name(), "object")


def main() raises:
    print("=" * 60)
    print("test_value.mojo")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
