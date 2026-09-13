# Tests for copy-on-write mutation via OwnedValue.
#
# These tests pin down the mutation contract:
#   1. `Value.set` / `Value.append` mutate in place and the change is
#      visible through every read API (`raw_json`, `string_value`, etc.).
#   2. The OwnedValue round-trip preserves all values that were
#      previously parsed -- it does not silently drop sibling keys or
#      reorder them within an object.
#   3. `Value.set_at(pointer, value)` propagates a mutation through the
#      full parent chain. Chaining does NOT: `__getitem__` hands back an
#      independent value, so `doc["a"].set("b", v)` edits a child `doc`
#      no longer shares and leaves `doc` alone. `test_chained_*` below
#      pins that, because the package docstring used to claim the
#      opposite.
#   4. The patch convenience pattern used by `json/patch.mojo`
#      (read parent, mutate, write parent back) keeps working.
#   5. `Value.object()` / `Value.array()` build the same JSON as the
#      `loads("{}")` route they replace, and a value is indistinguishable
#      through the public API regardless of which representation it
#      holds (tape view vs owned tree). See `json/value/value.mojo`.

from std.testing import assert_equal, assert_true, assert_false, TestSuite

from json import dumps, loads, Value, Null


# ---------------------------------------------------------------------------
# Top-level mutation.
# ---------------------------------------------------------------------------


def test_object_set_new_key_propagates() raises:
    """Setting a brand-new key on an object reflects in raw_json and reads."""
    var obj = loads('{"name":"Alice"}')
    obj.set("age", Value(Int64(30)))
    assert_true(obj.is_object())
    assert_equal(obj.object_count(), 2)
    var keys = obj.object_keys()
    assert_equal(len(keys), 2)
    assert_equal(obj["name"].string_value(), "Alice")
    assert_equal(obj["age"].int_value(), 30)


def test_object_set_update_key() raises:
    """Updating an existing key keeps the value count and updates the read."""
    var obj = loads('{"name":"Alice","age":30}')
    obj.set("name", Value("Bob"))
    assert_equal(obj.object_count(), 2)
    assert_equal(obj["name"].string_value(), "Bob")
    assert_equal(obj["age"].int_value(), 30)


def test_array_set_index() raises:
    """Replacing an element by index reflects in raw_json and reads."""
    var arr = loads("[1,2,3]")
    arr.set(1, Value(Int64(20)))
    assert_equal(arr.array_count(), 3)
    var items = arr.array_items()
    assert_equal(items[0].int_value(), 1)
    assert_equal(items[1].int_value(), 20)
    assert_equal(items[2].int_value(), 3)


def test_array_append() raises:
    """Append grows the array and the new value is observable."""
    var arr = loads("[1,2]")
    arr.append(Value(Int64(3)))
    assert_equal(arr.array_count(), 3)
    var items = arr.array_items()
    assert_equal(items[0].int_value(), 1)
    assert_equal(items[1].int_value(), 2)
    assert_equal(items[2].int_value(), 3)


def test_array_append_to_empty() raises:
    """Append to an empty array works correctly."""
    var arr = loads("[]")
    arr.append(Value(True))
    assert_equal(arr.array_count(), 1)
    assert_equal(arr.array_items()[0].bool_value(), True)


def test_object_set_on_empty() raises:
    """Setting on an empty object works correctly."""
    var obj = loads("{}")
    obj.set("first", Value(Int64(1)))
    assert_equal(obj.object_count(), 1)
    assert_equal(obj["first"].int_value(), 1)


# ---------------------------------------------------------------------------
# OwnedValue round-trip preserves all sibling values.
# ---------------------------------------------------------------------------


def test_object_set_preserves_other_keys() raises:
    """Mutating one key in an object doesn't lose any sibling keys."""
    var obj = loads('{"a":1,"b":2,"c":3,"d":4}')
    obj.set("b", Value(Int64(20)))
    assert_equal(obj.object_count(), 4)
    assert_equal(obj["a"].int_value(), 1)
    assert_equal(obj["b"].int_value(), 20)
    assert_equal(obj["c"].int_value(), 3)
    assert_equal(obj["d"].int_value(), 4)


def test_array_set_preserves_neighbors() raises:
    """Mutating one index in an array doesn't disturb its neighbors."""
    var arr = loads("[10,20,30,40,50]")
    arr.set(2, Value(Int64(99)))
    var items = arr.array_items()
    assert_equal(items[0].int_value(), 10)
    assert_equal(items[1].int_value(), 20)
    assert_equal(items[2].int_value(), 99)
    assert_equal(items[3].int_value(), 40)
    assert_equal(items[4].int_value(), 50)


def test_object_with_nested_array_preserved() raises:
    """A nested array stays parseable after mutating a sibling key."""
    var obj = loads('{"name":"Ada","tags":[1,2,3]}')
    obj.set("name", Value("Bob"))
    assert_equal(obj["name"].string_value(), "Bob")
    var tags = obj["tags"]
    assert_true(tags.is_array())
    assert_equal(tags.array_count(), 3)
    var items = tags.array_items()
    assert_equal(items[0].int_value(), 1)
    assert_equal(items[1].int_value(), 2)
    assert_equal(items[2].int_value(), 3)


def test_object_with_nested_object_preserved() raises:
    """A nested object stays parseable after mutating a sibling key."""
    var obj = loads('{"user":{"name":"Ada","age":36},"flag":true}')
    obj.set("flag", Value(False))
    assert_equal(obj["flag"].bool_value(), False)
    var user = obj["user"]
    assert_true(user.is_object())
    assert_equal(user["name"].string_value(), "Ada")
    assert_equal(user["age"].int_value(), 36)


def test_set_value_with_string_containing_quotes() raises:
    """Setting a string value containing quotes round-trips through escapes."""
    var obj = loads('{"k":1}')
    obj.set("msg", Value('hello "world"'))
    assert_equal(obj["msg"].string_value(), 'hello "world"')
    assert_equal(obj["k"].int_value(), 1)


def test_append_complex_value() raises:
    """Appending a parsed object preserves its structure."""
    var arr = loads("[1,2]")
    var nested = loads('{"k":42}')
    arr.append(nested)
    assert_equal(arr.array_count(), 3)
    var items = arr.array_items()
    assert_equal(items[0].int_value(), 1)
    assert_equal(items[1].int_value(), 2)
    assert_true(items[2].is_object())
    assert_equal(items[2]["k"].int_value(), 42)


# ---------------------------------------------------------------------------
# set_at(pointer, value) -- nested mutation entry point.
# ---------------------------------------------------------------------------


def test_set_at_top_level_object_key() raises:
    """`set_at("/key", v)` is equivalent to `set("key", v)` at the root."""
    var obj = loads('{"a":1,"b":2}')
    obj.set_at("/a", Value(Int64(100)))
    assert_equal(obj["a"].int_value(), 100)
    assert_equal(obj["b"].int_value(), 2)


def test_set_at_nested_object_key() raises:
    """A pointer two levels deep mutates the correct leaf."""
    var doc = loads('{"a":{"b":1,"c":2}}')
    doc.set_at("/a/b", Value(Int64(99)))
    var a = doc["a"]
    assert_equal(a["b"].int_value(), 99)
    assert_equal(a["c"].int_value(), 2)


def test_set_at_three_levels_deep() raises:
    """Three-level nested mutation propagates all the way up."""
    var doc = loads('{"a":{"b":{"c":1}}}')
    doc.set_at("/a/b/c", Value(Int64(42)))
    var leaf = doc["a"]["b"]["c"]
    assert_equal(leaf.int_value(), 42)


def test_set_at_array_index() raises:
    """A pointer with an array index mutates the indexed element."""
    var doc = loads('{"items":[10,20,30]}')
    doc.set_at("/items/1", Value(Int64(200)))
    var items = doc["items"].array_items()
    assert_equal(items[0].int_value(), 10)
    assert_equal(items[1].int_value(), 200)
    assert_equal(items[2].int_value(), 30)


def test_set_at_inserts_new_object_key() raises:
    """`set_at` on a missing leaf key under an existing object inserts it."""
    var doc = loads('{"a":{"b":1}}')
    doc.set_at("/a/c", Value(Int64(2)))
    var a = doc["a"]
    assert_equal(a.object_count(), 2)
    assert_equal(a["b"].int_value(), 1)
    assert_equal(a["c"].int_value(), 2)


def test_set_at_empty_pointer_replaces_root() raises:
    """An empty pointer replaces the entire document."""
    var doc = loads('{"a":1}')
    var replacement = loads("[1,2,3]")
    doc.set_at("", replacement)
    assert_true(doc.is_array())
    assert_equal(doc.array_count(), 3)


def test_set_at_path_does_not_exist_raises() raises:
    """Pointer through a missing intermediate path raises an error."""
    var doc = loads('{"a":{"b":1}}')
    var raised = False
    try:
        doc.set_at("/a/missing/leaf", Value(Int64(0)))
    except:
        raised = True
    assert_true(raised)


def test_set_at_through_primitive_raises() raises:
    """Pointer that descends into a primitive value raises an error."""
    var doc = loads('{"a":1}')
    var raised = False
    try:
        doc.set_at("/a/b", Value(Int64(0)))
    except:
        raised = True
    assert_true(raised)


# ---------------------------------------------------------------------------
# The patch.mojo / jsonpath.mojo workaround pattern still works.
# ---------------------------------------------------------------------------


def test_read_modify_write_parent_pattern() raises:
    """The 'fetch parent, mutate, set_at parent' pattern works end-to-end.

    json/patch.mojo uses this idiom heavily; if `set_at` regresses it,
    every JSON Patch operation breaks.
    """
    var doc = loads('{"users":[{"name":"Ada"}]}')
    var parent = doc.at("/users/0")
    parent.set("age", Value(Int64(36)))
    doc.set_at("/users/0", parent)
    var leaf = doc["users"].array_items()[0].copy()
    assert_equal(leaf["name"].string_value(), "Ada")
    assert_equal(leaf["age"].int_value(), 36)


def test_multiple_sequential_mutations() raises:
    """Many mutations in sequence each leave the document in a consistent state.
    """
    var doc = loads('{"counter":0}')
    for i in range(5):
        doc.set("counter", Value(Int64(i)))
    assert_equal(doc["counter"].int_value(), 4)


def test_append_multiple_times() raises:
    """Repeated append grows the array without losing earlier values."""
    var arr = loads("[]")
    for i in range(4):
        arr.append(Value(Int64(i)))
    assert_equal(arr.array_count(), 4)
    var items = arr.array_items()
    assert_equal(items[0].int_value(), 0)
    assert_equal(items[1].int_value(), 1)
    assert_equal(items[2].int_value(), 2)
    assert_equal(items[3].int_value(), 3)


# ---------------------------------------------------------------------------
# Construction factories, and representation transparency.
#
# `Value.object()` / `Value.array()` produce the owned representation;
# `loads(...)` produces a tape view. Every test below asserts the two are
# indistinguishable through the public API -- that is the whole contract
# of making the representation an implementation detail.
# ---------------------------------------------------------------------------


def test_object_factory_builds_empty_object() raises:
    """`Value.object()` is an empty JSON object, no parser involved."""
    var o = Value.object()
    assert_true(o.is_object())
    assert_false(o.is_array())
    assert_equal(o.object_count(), 0)
    assert_equal(dumps(o), "{}")


def test_array_factory_builds_empty_array() raises:
    """`Value.array()` is an empty JSON array."""
    var a = Value.array()
    assert_true(a.is_array())
    assert_false(a.is_object())
    assert_equal(a.array_count(), 0)
    assert_equal(dumps(a), "[]")


def test_factory_matches_loads_for_object() raises:
    """Building via the factory and via `loads("{}")` agree byte for byte."""
    var built = Value.object()
    built.set("name", Value("Ada"))
    built.set("age", Value(Int64(36)))

    var parsed = loads("{}")
    parsed.set("name", Value("Ada"))
    parsed.set("age", Value(Int64(36)))

    assert_equal(dumps(built), dumps(parsed))
    assert_equal(dumps(built), '{"name":"Ada","age":36}')


def test_factory_matches_loads_for_array() raises:
    """Same for arrays, including insertion order."""
    var built = Value.array()
    var parsed = loads("[]")
    for i in range(5):
        built.append(Value(Int64(i)))
        parsed.append(Value(Int64(i)))
    assert_equal(dumps(built), dumps(parsed))
    assert_equal(dumps(built), "[0,1,2,3,4]")


def test_factory_nested_structure_round_trips() raises:
    """A hand-built nested tree re-parses to the same thing."""
    var doc = Value.object()
    doc.set("id", Value("doc-1"))
    var meta = Value.object()
    meta.set("region", Value("us-east-1"))
    meta.set("version", Value(Int64(3)))
    doc.set("meta", meta)
    var items = Value.array()
    for i in range(3):
        var it = Value.object()
        it.set("sku", Value("sku-" + String(i)))
        it.set("qty", Value(Int64(i * 2)))
        items.append(it)
    doc.set("items", items)

    var text = dumps(doc)
    var back = loads(text)
    assert_equal(dumps(back), text)
    assert_equal(back["id"].string_value(), "doc-1")
    assert_equal(back["meta"]["region"].string_value(), "us-east-1")
    assert_equal(back["meta"]["version"].int_value(), 3)
    assert_equal(back["items"].array_count(), 3)
    assert_equal(back["items"][2]["sku"].string_value(), "sku-2")
    assert_equal(back["items"][2]["qty"].int_value(), 4)


def test_reads_agree_across_representations() raises:
    """Every read accessor gives the same answer either way."""
    var built = Value.object()
    built.set("s", Value("txt"))
    built.set("i", Value(Int64(-7)))
    built.set("f", Value(Float64(1.5)))
    built.set("b", Value(True))
    built.set("n", Value(Null()))
    var parsed = loads(dumps(built))

    assert_equal(built.object_count(), parsed.object_count())
    var bk = built.object_keys()
    var pk = parsed.object_keys()
    assert_equal(len(bk), len(pk))
    for i in range(len(bk)):
        assert_equal(bk[i], pk[i])
    for k in ["s", "i", "f", "b", "n"]:
        var a = built[k]
        var b = parsed[k]
        assert_equal(a.is_string(), b.is_string())
        assert_equal(a.is_int(), b.is_int())
        assert_equal(a.is_float(), b.is_float())
        assert_equal(a.is_bool(), b.is_bool())
        assert_equal(a.is_null(), b.is_null())
        assert_equal(a.is_number(), b.is_number())
        assert_equal(dumps(a), dumps(b))
    assert_equal(built["s"].string_value(), "txt")
    assert_equal(built["i"].int_value(), -7)
    assert_equal(built["f"].float_value(), 1.5)
    assert_equal(built["b"].bool_value(), True)
    assert_true(built["n"].is_null())


def test_object_items_agree_across_representations() raises:
    """`object_items` / `array_items` work on an owned tree too."""
    var built = Value.object()
    built.set("a", Value(Int64(1)))
    built.set("b", Value(Int64(2)))
    var pairs = built.object_items()
    assert_equal(len(pairs), 2)
    assert_equal(pairs[0][0], "a")
    assert_equal(pairs[0][1].int_value(), 1)
    assert_equal(pairs[1][0], "b")
    assert_equal(pairs[1][1].int_value(), 2)

    var arr = Value.array()
    arr.append(Value("x"))
    arr.append(Value("y"))
    var items = arr.array_items()
    assert_equal(len(items), 2)
    assert_equal(items[0].string_value(), "x")
    assert_equal(items[1].string_value(), "y")


def test_factory_raw_json_and_str_agree() raises:
    """`raw_json()` and `__str__` serialize an owned tree directly."""
    var o = Value.object()
    o.set("k", Value(Int64(1)))
    assert_equal(o.raw_json(), '{"k":1}')
    assert_equal(String(o), '{"k":1}')
    assert_equal(o.raw_member("k"), "1")


def test_factory_copy_is_independent() raises:
    """Copying an owned value gives value semantics, not aliasing."""
    var a = Value.array()
    a.append(Value(Int64(1)))
    var b = a.copy()
    b.append(Value(Int64(2)))
    assert_equal(a.array_count(), 1)
    assert_equal(b.array_count(), 2)


def test_factory_set_at_and_mutation_after_build() raises:
    """`set_at` works on a hand-built tree, not just a parsed one."""
    var doc = Value.object()
    var inner = Value.object()
    inner.set("b", Value(Int64(1)))
    doc.set("a", inner)
    doc.set_at("/a/b", Value(Int64(42)))
    assert_equal(doc["a"]["b"].int_value(), 42)
    doc.set_at("/a/c", Value("new"))
    assert_equal(doc["a"]["c"].string_value(), "new")
    assert_equal(doc["a"].object_count(), 2)


def test_mutating_parsed_value_then_reading_all_paths() raises:
    """A parsed value converts on first mutation and stays readable.

    Pins the conversion boundary: mutate a tape-backed value, then
    exercise the read surface that now has to answer from the owned tree.
    """
    var doc = loads('{"a":[1,2],"b":{"c":"z"},"d":true}')
    doc.set("e", Value(Int64(9)))
    assert_equal(doc.object_count(), 4)
    assert_equal(doc["a"].array_count(), 2)
    assert_equal(doc["a"][1].int_value(), 2)
    assert_equal(doc["b"]["c"].string_value(), "z")
    assert_equal(doc["d"].bool_value(), True)
    assert_equal(doc["e"].int_value(), 9)
    assert_equal(len(doc.object_keys()), 4)
    assert_equal(dumps(loads(dumps(doc))), dumps(doc))


def test_deeply_nested_build_stays_correct() raises:
    """Nesting depth is handled by the owned representation."""
    var cur = Value.object()
    cur.set("v", Value(Int64(0)))
    for i in range(1, 12):
        var parent = Value.object()
        parent.set("v", Value(Int64(i)))
        parent.set("child", cur)
        cur = parent^
    var text = dumps(cur)
    var back = loads(text)
    assert_equal(dumps(back), text)
    var walk = back.copy()
    for i in range(11, 0, -1):
        assert_equal(walk["v"].int_value(), Int64(i))
        var child = walk["child"]
        walk = child^
    assert_equal(walk["v"].int_value(), 0)


# ---------------------------------------------------------------------------
# What chaining does and does not do (bug 2).
#
# `json/__init__.mojo` used to promise that a mutation through a
# chained subscript was observed by the parent. It never was: every
# mutator calls `_to_owned_in_place`, which detaches the child from the
# shared document. These tests pin the behaviour the documentation now
# describes, so a future change to either one has to face the other.
# ---------------------------------------------------------------------------


def test_chained_subscript_mutation_does_not_reach_the_parent() raises:
    """`doc["a"].set(...)` leaves `doc` untouched, and must keep saying so."""
    var doc = loads('{"a":{"b":1}}')
    var child = doc["a"]
    child.set("b", Value(Int64(42)))
    assert_equal(child["b"].int_value(), 42)
    assert_equal(
        doc["a"]["b"].int_value(), 1, "the parent must still read the old value"
    )
    assert_equal(dumps(doc), '{"a":{"b":1}}')


def test_chained_temporary_mutation_is_discarded() raises:
    """The one-liner shape drops the mutation on the floor entirely."""
    var doc = loads('{"a":{"b":1}}')
    var temp = doc["a"]
    temp.set("b", Value(Int64(99)))
    assert_equal(dumps(doc), '{"a":{"b":1}}')


def test_set_at_is_how_a_nested_write_reaches_the_parent() raises:
    """`set_at(pointer, value)` is the spelling that does propagate."""
    var doc = loads('{"a":{"b":1},"c":2}')
    doc.set_at("/a/b", Value(Int64(42)))
    assert_equal(doc["a"]["b"].int_value(), 42)
    assert_equal(doc["c"].int_value(), 2, "siblings are left alone")
    assert_equal(dumps(doc), '{"a":{"b":42},"c":2}')


def test_chained_array_mutation_does_not_reach_the_parent() raises:
    """The same holds for an array element reached by subscript."""
    var doc = loads('{"a":[1,2,3]}')
    var arr = doc["a"]
    arr.set(0, Value(Int64(9)))
    assert_equal(arr[0].int_value(), 9)
    assert_equal(doc["a"][0].int_value(), 1)
    doc.set_at("/a/0", Value(Int64(9)))
    assert_equal(doc["a"][0].int_value(), 9)


# ---------------------------------------------------------------------------
# Subscript assignment.
# ---------------------------------------------------------------------------


def test_setitem_on_object() raises:
    """`obj["k"] = v` adds and updates members."""
    var obj = loads('{"name":"Alice"}')
    obj["age"] = Value(Int64(30))
    obj["name"] = Value("Bob")
    assert_equal(obj.object_count(), 2)
    assert_equal(obj["name"].string_value(), "Bob")
    assert_equal(obj["age"].int_value(), 30)


def test_setitem_on_array_including_negative() raises:
    """`arr[0] = v` and `arr[-1] = v` replace elements."""
    var arr = loads("[1,2,3]")
    arr[0] = Value(Int64(10))
    arr[-1] = Value(Int64(30))
    assert_equal(dumps(arr), "[10,2,30]")


def test_setitem_out_of_range_raises() raises:
    """Assignment replaces; it does not grow the array."""
    var arr = loads("[1]")
    var raised = False
    try:
        arr[5] = Value(Int64(1))
    except:
        raised = True
    assert_true(raised, "an index past the end must raise")


def test_setitem_on_owned_tree() raises:
    """Subscript assignment works on a hand-built value too."""
    var obj = Value.object()
    obj["a"] = Value(Int64(1))
    var arr = Value.array()
    arr.append(Value(Int64(0)))
    arr[0] = Value("replaced")
    obj["list"] = arr^
    assert_equal(dumps(obj), '{"a":1,"list":["replaced"]}')


# ---------------------------------------------------------------------------
# Removal.
# ---------------------------------------------------------------------------


def test_remove_object_member() raises:
    """`remove(key)` drops a member and keeps the order of the rest."""
    var obj = loads('{"a":1,"b":2,"c":3}')
    obj.remove("b")
    assert_equal(obj.object_count(), 2)
    assert_equal(dumps(obj), '{"a":1,"c":3}')
    assert_false("b" in obj, "the member is gone")


def test_remove_missing_member_raises() raises:
    """Removing what is not there is an error, not a silent no-op."""
    var obj = loads('{"a":1}')
    var raised = False
    try:
        obj.remove("zz")
    except:
        raised = True
    assert_true(raised, "removing an absent member must raise")
    assert_equal(obj.object_count(), 1)


def test_remove_array_element() raises:
    """`remove(index)` shifts the later elements down."""
    var arr = loads("[1,2,3,4]")
    arr.remove(1)
    assert_equal(dumps(arr), "[1,3,4]")
    arr.remove(-1)
    assert_equal(dumps(arr), "[1,3]")


def test_pop_object_member_returns_it() raises:
    """`pop(key)` hands back the value it removed."""
    var obj = loads('{"a":{"deep":[1,2]},"b":2}')
    var taken = obj.pop("a")
    assert_equal(dumps(taken), '{"deep":[1,2]}')
    assert_equal(dumps(obj), '{"b":2}')


def test_pop_array_element_and_last() raises:
    """`pop(index)`, `pop(-1)` and the no-argument `pop()` all work."""
    var arr = loads('[1,"two",3]')
    assert_equal(arr.pop(1).string_value(), "two")
    assert_equal(dumps(arr), "[1,3]")
    assert_equal(arr.pop().int_value(), 3)
    assert_equal(dumps(arr), "[1]")
    assert_equal(arr.pop(-1).int_value(), 1)
    assert_equal(dumps(arr), "[]")

    var raised = False
    try:
        _ = arr.pop()
    except:
        raised = True
    assert_true(raised, "popping an empty array must raise")


def test_remove_then_rebuild_round_trips() raises:
    """A value survives removal and reserializes to valid JSON."""
    var doc = loads('{"keep":[1,2,3],"drop":{"x":1}}')
    doc.remove("drop")
    # A read of the detached child changes nothing.
    _ = doc["keep"].__str__()
    assert_equal(dumps(loads(dumps(doc))), dumps(doc))
    assert_equal(dumps(doc), '{"keep":[1,2,3]}')


# ---------------------------------------------------------------------------
# Ergonomics on a mutated value.
# ---------------------------------------------------------------------------


def test_iteration_and_len_after_mutation() raises:
    """The lazy iterators read the owned tree a mutation left behind."""
    var doc = loads('{"a":1}')
    doc["b"] = Value(Int64(2))
    assert_equal(len(doc), 2)
    var names = String()
    for key in doc.keys():
        names += key
    assert_equal(names, "ab")

    var arr = loads("[1]")
    arr.append(Value(Int64(2)))
    assert_equal(len(arr), 2)
    var total = Int64(0)
    for item in arr:
        total += item.int_value()
    assert_equal(total, 3)


def test_equality_survives_the_representation_switch() raises:
    """A mutated value equals the parse of what it serializes to."""
    var doc = loads('{"a":{"x":1},"b":[1,2]}')
    doc.set_at("/a/x", Value(Int64(5)))
    assert_true(doc == loads('{"b":[1,2],"a":{"x":5}}'), "order-insensitive")
    assert_equal(doc.hash_u64(), loads('{"b":[1,2],"a":{"x":5}}').hash_u64())


def main() raises:
    print("=" * 60)
    print("test_value_mutation.mojo")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
