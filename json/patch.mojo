# json - JSON Patch (RFC 6902) and JSON Merge Patch (RFC 7396)
#
# JSON Patch: Apply a sequence of operations to a JSON document
# JSON Merge Patch: Merge two JSON documents together
#
# Operation paths are reference-token lists from `json/pointer.mojo`
# rather than strings sliced on the way past. The previous shape
# resolved a parent with one pointer parser and computed the final
# token with another, so `"/"`, `"//"` and `"/a/"` were validated
# against one location and written to a different one.

from std.collections import List

from .pointer import array_index, index_value, parse_pointer
from .value import Value, Null
from .parser import loads
from .serialize import dumps


# =============================================================================
# JSON Patch (RFC 6902)
# =============================================================================


def apply_patch(document: Value, patch: Value) raises -> Value:
    """Apply a JSON Patch (RFC 6902) to a document.

    The patch is a JSON array of operations. Each operation has:
    - "op": The operation ("add", "remove", "replace", "move", "copy", "test")
    - "path": JSON Pointer to the target location
    - "value": The value (for add, replace, test)
    - "from": Source path (for move, copy)

    Operations are applied in order, and the whole patch either
    succeeds or changes nothing: the document is never edited in
    place, so an operation that raises leaves the caller's value as it
    was. The failure names the operation's index, because a patch that
    only says "key not found" is hard to place in a long list.

    Args:
        document: The original JSON document.
        patch: Array of patch operations.

    Returns:
        New document with patches applied.

    Example:
        var doc = loads('{"name":"Alice"}')
        var patch = loads('[{"op":"add","path":"/age","value":30}]')
        var result = apply_patch(doc, patch)
        Result is `{"name":"Alice","age":30}`.
    """
    if not patch.is_array():
        raise Error("JSON Patch must be an array")

    var result = document.copy()
    var ops = patch.array_items()

    for i in range(len(ops)):
        try:
            result = _apply_operation(result, ops[i])
        except e:
            raise Error(
                "JSON Patch operation " + String(i) + " failed: " + String(e)
            )

    return result^


def _member(operation: Value, name: String) raises -> Value:
    """A required member of a patch operation."""
    try:
        return operation[name]
    except:
        raise Error("operation has no '" + name + "' member")


def _string_member(operation: Value, name: String) raises -> String:
    """A required member that RFC 6902 spells as a string.

    `Value.string_value` does not check the tag, so reading `path` off
    an operation that spells it as a number used to hand back whatever
    the number's payload looked like as an offset into the input. An
    empty result then meant "the whole document", and the operation
    silently replaced it.
    """
    var member = _member(operation, name)
    if not member.is_string():
        raise Error("'" + name + "' must be a string")
    return member.string_value()


def _reject_duplicate_members(operation: Value) raises:
    """RFC 6902 appendix A.13: an operation names each member once.

    JSON itself tolerates a repeated name and this library keeps both,
    so an operation carrying two `op` members would otherwise be read
    as whichever one the lookup reached first and applied as if it
    were well formed.
    """
    var members = operation.object_items()
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if members[i][0] == members[j][0]:
                raise Error(
                    "operation names '" + members[i][0] + "' more than once"
                )


def _apply_operation(document: Value, operation: Value) raises -> Value:
    """Apply a single patch operation."""
    if not operation.is_object():
        raise Error("patch operation must be an object")
    _reject_duplicate_members(operation)

    var op_type = _string_member(operation, "op")
    var path = _string_member(operation, "path")
    var tokens = parse_pointer(path)

    if op_type == "add":
        return _add_at(document, tokens, 0, _member(operation, "value"))
    elif op_type == "remove":
        if len(tokens) == 0:
            raise Error("cannot remove the whole document")
        return _remove_at(document, tokens, 0)
    elif op_type == "replace":
        if len(tokens) == 0:
            return _member(operation, "value").copy()
        return _replace_at(document, tokens, 0, _member(operation, "value"))
    elif op_type == "move":
        var from_tokens = parse_pointer(_string_member(operation, "from"))
        return _move(document, from_tokens, tokens)
    elif op_type == "copy":
        var from_tokens = parse_pointer(_string_member(operation, "from"))
        var source = _resolve(document, from_tokens)
        return _add_at(document, tokens, 0, source)
    elif op_type == "test":
        var expected = _member(operation, "value")
        var actual = _resolve(document, tokens)
        if actual != expected:
            raise Error("test failed: value at '" + path + "' does not match")
        return document.copy()
    else:
        raise Error("unknown patch operation: " + op_type)


def _resolve(document: Value, tokens: List[String]) raises -> Value:
    """The value named by `tokens`, or an error naming what is missing."""
    var current = document.copy()
    for i in range(len(tokens)):
        var token = tokens[i]
        if current.is_object():
            current = current[token]
        elif current.is_array():
            current = current[array_index(token, current.array_count())]
        else:
            raise Error(
                "'" + token + "' names a member of a value that has none"
            )
    return current^


def _add_at(
    node: Value, tokens: List[String], depth: Int, value: Value
) raises -> Value:
    """RFC 6902 `add`, rebuilding the containers along the way.

    Whether the final token is an array index or an object member is
    decided by the container's runtime type, not by how the token is
    spelled, which is what RFC 6902 section 4.1 requires.
    """
    if depth == len(tokens):
        return value.copy()

    var token = tokens[depth]
    var last = depth == len(tokens) - 1

    if node.is_object():
        if last:
            var out = node.copy()
            out.set(token, value)
            return out^
        var child = _add_at(node[token], tokens, depth + 1, value)
        var out = node.copy()
        out.set(token, child)
        return out^

    if node.is_array():
        var count = node.array_count()
        if last:
            var index: Int
            if token == "-":
                index = count
            else:
                index = index_value(token)
                if index > count:
                    raise Error(
                        "cannot add at index "
                        + token
                        + " of an array of length "
                        + String(count)
                    )
            return _array_insert(node, index, value)
        var index = array_index(token, count)
        var child = _add_at(node[index], tokens, depth + 1, value)
        var out = node.copy()
        out.set(index, child)
        return out^

    raise Error("cannot add '" + token + "' to a value that has no members")


def _remove_at(node: Value, tokens: List[String], depth: Int) raises -> Value:
    """RFC 6902 `remove`. The target must exist."""
    var token = tokens[depth]
    var last = depth == len(tokens) - 1

    if node.is_object():
        if last:
            return _remove_object_key(node, token, required=True)
        var child = _remove_at(node[token], tokens, depth + 1)
        var out = node.copy()
        out.set(token, child)
        return out^

    if node.is_array():
        var index = array_index(token, node.array_count())
        if last:
            return _array_remove(node, index)
        var child = _remove_at(node[index], tokens, depth + 1)
        var out = node.copy()
        out.set(index, child)
        return out^

    raise Error(
        "cannot remove '" + token + "' from a value that has no members"
    )


def _replace_at(
    node: Value, tokens: List[String], depth: Int, value: Value
) raises -> Value:
    """RFC 6902 `replace`. The target must already exist."""
    var token = tokens[depth]
    var last = depth == len(tokens) - 1

    if node.is_object():
        # Reading the member first is what makes replacing something
        # that is not there an error rather than an add.
        var existing = node[token]
        if last:
            var out = node.copy()
            out.set(token, value)
            return out^
        var child = _replace_at(existing, tokens, depth + 1, value)
        var out = node.copy()
        out.set(token, child)
        return out^

    if node.is_array():
        var index = array_index(token, node.array_count())
        if last:
            var out = node.copy()
            out.set(index, value)
            return out^
        var child = _replace_at(node[index], tokens, depth + 1, value)
        var out = node.copy()
        out.set(index, child)
        return out^

    raise Error("cannot replace '" + token + "' in a value that has no members")


def _move(
    document: Value, from_tokens: List[String], to_tokens: List[String]
) raises -> Value:
    """RFC 6902 `move`, which is a remove followed by an add."""
    if _same_location(from_tokens, to_tokens):
        # Removing and re-adding in one place is a no-op for an object
        # and, for an array, an identity insert. Saying so directly is
        # cheaper and cannot get the index arithmetic wrong.
        return document.copy()
    if _is_proper_prefix(from_tokens, to_tokens):
        raise Error("cannot move a value into one of its own children")
    var value = _resolve(document, from_tokens)
    if len(from_tokens) == 0:
        raise Error("cannot move the whole document")
    var without = _remove_at(document, from_tokens, 0)
    return _add_at(without, to_tokens, 0, value)


def _same_location(a: List[String], b: List[String]) -> Bool:
    """Whether two token lists name the same place."""
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _is_proper_prefix(a: List[String], b: List[String]) -> Bool:
    """Whether `a` names a strict ancestor of `b`."""
    if len(a) >= len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


# =============================================================================
# JSON Merge Patch (RFC 7396)
# =============================================================================


def merge_patch(target: Value, patch: Value) raises -> Value:
    """Apply a JSON Merge Patch (RFC 7396) to a document.

    Merge patch rules:
    - If patch is not an object, it replaces target entirely
    - null values in patch remove keys from target
    - Other values are recursively merged

    Args:
        target: The original document.
        patch: The merge patch to apply.

    Returns:
        New document with merge patch applied.

    Example:
        var target = loads('{"a":1,"b":2}')
        var patch = loads('{"b":null,"c":3}')
        var result = merge_patch(target, patch)
        Result is `{"a":1,"c":3}`.
    """
    if not patch.is_object():
        return patch.copy()

    var result: Value
    if target.is_object():
        result = target.copy()
    else:
        result = loads("{}")

    var patch_items = patch.object_items()
    for i in range(len(patch_items)):
        var key = patch_items[i][0]
        var value = patch_items[i][1].copy()

        if value.is_null():
            # Remove the key
            result = _remove_object_key(result, key, required=False)
        else:
            # Recursively merge
            var target_value: Value
            try:
                target_value = result[key].copy()
            except:
                target_value = Value(Null())

            var merged = merge_patch(target_value, value)
            result.set(key, merged)

    return result^


def create_merge_patch(source: Value, target: Value) raises -> Value:
    """Create a merge patch that transforms source into target.

    Args:
        source: The original document.
        target: The desired result.

    Returns:
        A merge patch that when applied to source produces target.

    Example:
        var source = loads('{"a":1,"b":2}')
        var target = loads('{"a":1,"c":3}')
        var patch = create_merge_patch(source, target)
        Result is `{"b":null,"c":3}`.
    """
    if not target.is_object():
        return target.copy()

    if not source.is_object():
        return target.copy()

    var patch = loads("{}")

    # Find removed and changed keys
    var source_items = source.object_items()
    for i in range(len(source_items)):
        var key = source_items[i][0]
        var source_val = source_items[i][1].copy()

        var target_has_key = False
        var target_val: Value
        try:
            target_val = target[key].copy()
            target_has_key = True
        except:
            target_val = Value(Null())

        if not target_has_key:
            # Key was removed
            patch.set(key, Value(Null()))
        elif source_val != target_val:
            # Key was changed
            if source_val.is_object() and target_val.is_object():
                var sub_patch = create_merge_patch(source_val, target_val)
                patch.set(key, sub_patch)
            else:
                patch.set(key, target_val)

    # Find added keys
    var target_items = target.object_items()
    for i in range(len(target_items)):
        var key = target_items[i][0]
        var target_val = target_items[i][1].copy()

        var source_has_key = False
        try:
            _ = source[key]
            source_has_key = True
        except:
            pass

        if not source_has_key:
            patch.set(key, target_val)

    return patch^


# =============================================================================
# Helper Functions
# =============================================================================


def _remove_object_key(
    obj: Value, key: String, *, required: Bool
) raises -> Value:
    """An object without `key`.

    Built member by member rather than by splicing JSON text back
    together, which is how an earlier version corrupted any document
    holding a key that needs escaping: the key went back into the text
    raw, and the re-parse either failed or read a different key.

    Args:
        obj: The object to copy.
        key: The member to leave out.
        required: Whether a missing member is an error. RFC 6902's
            `remove` says it is; RFC 7396's `null` says it is not.
    """
    var out = Value.object()
    var found = False
    var items = obj.object_items()
    for i in range(len(items)):
        if items[i][0] == key:
            found = True
            continue
        out.set(items[i][0], items[i][1])
    if required and not found:
        raise Error("cannot remove member '" + key + "', which is not present")
    return out^


def _array_insert(arr: Value, index: Int, value: Value) raises -> Value:
    """An array with `value` inserted before position `index`."""
    var out = Value.array()
    var items = arr.array_items()
    for i in range(len(items)):
        if i == index:
            out.append(value)
        out.append(items[i])
    if index >= len(items):
        out.append(value)
    return out^


def _array_remove(arr: Value, index: Int) raises -> Value:
    """An array without the element at `index`."""
    var out = Value.array()
    var items = arr.array_items()
    for i in range(len(items)):
        if i != index:
            out.append(items[i])
    return out^
