# Tests for JSON Patch (RFC 6902) and JSON Merge Patch (RFC 7396)

from std.testing import assert_equal, assert_true, TestSuite

from json import loads, dumps, apply_patch, merge_patch, create_merge_patch


# JSON Patch tests


def test_patch_add_new_key() raises:
    """Test adding a new key to an object."""
    var doc = loads('{"name":"Alice"}')
    var patch = loads('[{"op":"add","path":"/age","value":30}]')
    var result = apply_patch(doc, patch)
    assert_true(result["age"].int_value() == 30)


def test_patch_add_to_array() raises:
    """Test adding to an array."""
    var doc = loads('{"items":[1,2]}')
    var patch = loads('[{"op":"add","path":"/items/-","value":3}]')
    var result = apply_patch(doc, patch)
    assert_equal(result["items"].array_count(), 3)


def test_patch_add_array_middle() raises:
    """Test adding to middle of array."""
    var doc = loads("[1,2,3]")
    var patch = loads('[{"op":"add","path":"/1","value":99}]')
    var result = apply_patch(doc, patch)
    assert_equal(Int(result[1].int_value()), 99)


def test_patch_remove_key() raises:
    """Test removing a key."""
    var doc = loads('{"a":1,"b":2}')
    var patch = loads('[{"op":"remove","path":"/a"}]')
    var result = apply_patch(doc, patch)
    var caught = False
    try:
        _ = result["a"]
    except:
        caught = True
    assert_true(caught)


def test_patch_remove_array_element() raises:
    """Test removing array element."""
    var doc = loads("[1,2,3]")
    var patch = loads('[{"op":"remove","path":"/1"}]')
    var result = apply_patch(doc, patch)
    assert_equal(result.array_count(), 2)


def test_patch_replace() raises:
    """Test replacing a value."""
    var doc = loads('{"name":"Alice"}')
    var patch = loads('[{"op":"replace","path":"/name","value":"Bob"}]')
    var result = apply_patch(doc, patch)
    assert_equal(result["name"].string_value(), "Bob")


def test_patch_move() raises:
    """Test moving a value."""
    var doc = loads('{"a":1,"b":2}')
    var patch = loads('[{"op":"move","from":"/a","path":"/c"}]')
    var result = apply_patch(doc, patch)
    assert_equal(Int(result["c"].int_value()), 1)


def test_patch_copy() raises:
    """Test copying a value."""
    var doc = loads('{"a":1}')
    var patch = loads('[{"op":"copy","from":"/a","path":"/b"}]')
    var result = apply_patch(doc, patch)
    assert_equal(Int(result["a"].int_value()), 1)
    assert_equal(Int(result["b"].int_value()), 1)


def test_patch_test_pass() raises:
    """Test the test operation (passing)."""
    var doc = loads('{"a":1}')
    var patch = loads('[{"op":"test","path":"/a","value":1}]')
    var result = apply_patch(doc, patch)
    assert_true(result.is_object())


def test_patch_test_fail() raises:
    """Test the test operation (failing)."""
    var doc = loads('{"a":1}')
    var patch = loads('[{"op":"test","path":"/a","value":2}]')
    var caught = False
    try:
        _ = apply_patch(doc, patch)
    except:
        caught = True
    assert_true(caught)


def test_patch_multiple_ops() raises:
    """Test multiple operations."""
    var doc = loads('{"name":"Alice","age":25}')
    var patch = loads(
        '[{"op":"replace","path":"/name","value":"Bob"},{"op":"add","path":"/active","value":true}]'
    )
    var result = apply_patch(doc, patch)
    assert_equal(result["name"].string_value(), "Bob")
    assert_true(result["active"].bool_value())


# JSON Merge Patch tests


def test_merge_patch_add() raises:
    """Test merge patch adding a key."""
    var target = loads('{"a":1}')
    var patch = loads('{"b":2}')
    var result = merge_patch(target, patch)
    assert_equal(Int(result["a"].int_value()), 1)
    assert_equal(Int(result["b"].int_value()), 2)


def test_merge_patch_remove() raises:
    """Test merge patch removing a key (null)."""
    var target = loads('{"a":1,"b":2}')
    var patch = loads('{"b":null}')
    var result = merge_patch(target, patch)
    var caught = False
    try:
        _ = result["b"]
    except:
        caught = True
    assert_true(caught)


def test_merge_patch_replace() raises:
    """Test merge patch replacing a value."""
    var target = loads('{"a":1}')
    var patch = loads('{"a":2}')
    var result = merge_patch(target, patch)
    assert_equal(Int(result["a"].int_value()), 2)


def test_merge_patch_nested() raises:
    """Test nested merge patch."""
    var target = loads('{"a":{"b":1}}')
    var patch = loads('{"a":{"c":2}}')
    var result = merge_patch(target, patch)
    assert_equal(Int(result["a"]["b"].int_value()), 1)
    assert_equal(Int(result["a"]["c"].int_value()), 2)


def test_merge_patch_replace_object() raises:
    """Test merge patch replacing entire object."""
    var target = loads('{"a":1}')
    var patch = loads("[1,2,3]")
    var result = merge_patch(target, patch)
    assert_true(result.is_array())


def test_create_merge_patch() raises:
    """Test creating a merge patch."""
    var source = loads('{"a":1,"b":2}')
    var target = loads('{"a":1,"c":3}')
    var patch = create_merge_patch(source, target)
    assert_true(patch["b"].is_null())
    assert_equal(Int(patch["c"].int_value()), 3)


def _must_raise(doc: String, patch: String) raises -> String:
    """Apply a patch that must fail, and return the message.

    Also asserts the document the caller passed in is untouched, which
    is the atomicity requirement of RFC 6902 section 5.
    """
    var before = loads(doc)
    var text = dumps(before)
    var raised = String("")
    try:
        _ = apply_patch(before, loads(patch))
    except e:
        raised = String(e)
    assert_true(raised != "", "expected the patch to fail: " + patch)
    assert_equal(dumps(before), text)
    return raised^


def test_patch_rejects_a_non_string_path() raises:
    """`path` spelled as a number used to read as the empty pointer.

    That made the operation replace the whole document instead of
    failing, which is the worst thing an unvalidated field can do.
    """
    var message = _must_raise('{"a":1}', '[{"op":"add","path":123,"value":2}]')
    assert_true("'path' must be a string" in message, message)


def test_patch_rejects_a_missing_value() raises:
    var message = _must_raise('{"a":1}', '[{"op":"add","path":"/b"}]')
    assert_true("has no 'value' member" in message, message)


def test_patch_rejects_an_unrooted_pointer() raises:
    _ = _must_raise('{"a":1}', '[{"op":"add","path":"a","value":2}]')


def test_patch_rejects_an_invalid_pointer_escape() raises:
    _ = _must_raise('{"a":1}', '[{"op":"add","path":"/~2","value":2}]')


def test_patch_rejects_a_leading_zero_index() raises:
    _ = _must_raise('{"a":[1,2]}', '[{"op":"remove","path":"/a/01"}]')


def test_patch_rejects_dash_outside_add() raises:
    """`-` names a position no existing element occupies."""
    _ = _must_raise('{"a":[1,2]}', '[{"op":"remove","path":"/a/-"}]')


def test_patch_add_at_the_end_of_an_array() raises:
    """Index equal to the length is legal; one past it is not."""
    var doc = loads('{"a":[1,2]}')
    var out = apply_patch(doc, loads('[{"op":"add","path":"/a/2","value":9}]'))
    assert_equal(dumps(out), '{"a":[1,2,9]}')
    _ = _must_raise('{"a":[1,2]}', '[{"op":"add","path":"/a/3","value":9}]')


def test_patch_remove_requires_the_member_to_exist() raises:
    var message = _must_raise('{"a":1}', '[{"op":"remove","path":"/nope"}]')
    assert_true("not present" in message, message)


def test_patch_replace_requires_the_member_to_exist() raises:
    _ = _must_raise('{"a":1}', '[{"op":"replace","path":"/nope","value":2}]')


def test_patch_move_into_own_child_is_an_error() raises:
    var message = _must_raise(
        '{"a":{"b":1}}', '[{"op":"move","from":"/a","path":"/a/c"}]'
    )
    assert_true("own children" in message, message)


def test_patch_test_compares_numbers_by_value() raises:
    """RFC 6902 section 4.6: `1` and `1.0` are the same number."""
    var doc = loads('{"a":1}')
    var out = apply_patch(doc, loads('[{"op":"test","path":"/a","value":1.0}]'))
    assert_equal(dumps(out), '{"a":1}')


def test_patch_test_ignores_member_order() raises:
    var doc = loads('{"a":{"x":1,"y":2}}')
    var patch = loads('[{"op":"test","path":"/a","value":{"y":2,"x":1}}]')
    assert_equal(dumps(apply_patch(doc, patch)), '{"a":{"x":1,"y":2}}')


def test_patch_is_atomic_and_names_the_operation() raises:
    var message = _must_raise(
        '{"a":1,"b":2}',
        '[{"op":"remove","path":"/a"},{"op":"remove","path":"/zzz"}]',
    )
    assert_true("operation 1" in message, message)


def test_patch_keeps_keys_that_need_escaping() raises:
    """Rebuilding a container used to splice keys back in unescaped."""
    var doc = loads('{"a\\"b":1,"x":2}')
    var out = apply_patch(doc, loads('[{"op":"remove","path":"/x"}]'))
    assert_equal(dumps(out), '{"a\\"b":1}')


def test_patch_addresses_the_empty_key() raises:
    """The pointer `/` names the member whose key is the empty string."""
    var doc = loads('{"":5}')
    var out = apply_patch(doc, loads('[{"op":"replace","path":"/","value":6}]'))
    assert_equal(dumps(out), '{"":6}')


def test_patch_replaces_the_whole_document() raises:
    var doc = loads('{"a":1}')
    var out = apply_patch(doc, loads('[{"op":"add","path":"","value":[1,2]}]'))
    assert_equal(dumps(out), "[1,2]")


def test_merge_patch_null_for_an_absent_key_is_a_no_op() raises:
    """RFC 7396 removes a member that is there and ignores one that is not."""
    var out = merge_patch(loads('{"a":1}'), loads('{"b":null}'))
    assert_equal(dumps(out), '{"a":1}')


def test_merge_patch_keeps_keys_that_need_escaping() raises:
    var out = merge_patch(loads('{"a\\"b":1,"x":2}'), loads('{"x":null}'))
    assert_equal(dumps(out), '{"a\\"b":1}')


def main() raises:
    print("=" * 60)
    print("test_patch.mojo - JSON Patch Tests")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
