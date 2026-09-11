# json - Conversion between `Value` (tape view) and `OwnedValue` (tree).
#
# The owned tree itself lives in `node.mojo`; this module owns the
# translation in both directions plus the JSON-Pointer mutation walk:
#
#   Document tape --(_view_to_owned)--> OwnedValue
#   OwnedValue    --(_owned_to_tape)--> Document tape
#   raw JSON      --(_parse_owned_value)--> OwnedValue
#   OwnedValue    --(_owned_to_json)--> raw JSON
#
# Each conversion is O(N) in the subtree, so callers must not run one
# per mutation. `Value` keeps the owned tree live across mutations and
# converts at most once, on the first write to a tape-backed value --
# doing it per mutation is what previously made building a tree O(N^2).

from std.collections import List
from std.memory import ArcPointer

from .node import (
    OwnedValue,
    OWNED_NULL,
    OWNED_BOOL,
    OWNED_INT,
    OWNED_FLOAT,
    OWNED_STRING,
    OWNED_ARRAY,
    OWNED_OBJECT,
)
from .raw_ops import escape_json_string
from .value import Value, Null, make_view_value
from ..writer import JsonWriter
from ..document import (
    Document,
    pack_tape_entry,
    pack_pair,
    TAPE_TAG_NULL,
    TAPE_TAG_BOOL,
    TAPE_TAG_INT,
    TAPE_TAG_FLOAT,
    TAPE_TAG_STRING,
    TAPE_TAG_STRING_OWNED,
    TAPE_TAG_ARRAY,
    TAPE_TAG_OBJECT,
    TAPE_TAG_KEY,
)
from ..unicode import unescape_json_string, unescape_json_string_span


# ---------------------------------------------------------------------------
# Value <-> OwnedValue conversion
# ---------------------------------------------------------------------------


def _value_to_owned(v: Value) raises -> OwnedValue:
    """Convert a `Value` to an `OwnedValue` tree.

    A value already holding an owned tree hands back a copy. A
    tape-backed view walks `Document.tape` directly via
    `_view_to_owned` -- no JSON serialization, no parsing.
    """
    if not v._is_view():
        return v._owned.copy()
    if v.is_null():
        return OwnedValue.make_null()
    if v.is_bool():
        return OwnedValue.make_bool(v.bool_value())
    if v.is_int():
        return OwnedValue.make_int(v.int_value())
    if v.is_float():
        return OwnedValue.make_float(v.float_value())
    if v.is_string():
        return OwnedValue.make_string(v.string_value())
    if v.is_array() or v.is_object():
        return _view_to_owned(v._doc.value(), v._tape_idx)
    raise Error("Unknown Value kind in _value_to_owned")


def _view_to_owned(
    doc: ArcPointer[Document], tape_idx: Int
) raises -> OwnedValue:
    """Walk a tape entry into a fresh `OwnedValue` tree.

    This is the COW materialization path for tape-backed Values: no
    raw JSON is produced or parsed, we just translate tape entries
    one-for-one into `OwnedValue.make_*` constructors.

    For arrays / objects we use `Document.get_count` and
    `Document.get_child_start` to find children; KEY entries
    (between OBJECT pairs) are read out of `key_pool` via
    `Document.get_key`.
    """
    ref d = doc[]
    var tag = d.get_tag(tape_idx)
    if tag == TAPE_TAG_NULL:
        return OwnedValue.make_null()
    if tag == TAPE_TAG_BOOL:
        return OwnedValue.make_bool(d.get_bool(tape_idx))
    if tag == TAPE_TAG_INT:
        return OwnedValue.make_int(d.get_int(tape_idx))
    if tag == TAPE_TAG_FLOAT:
        return OwnedValue.make_float(d.get_float(tape_idx))
    if tag == TAPE_TAG_STRING or tag == TAPE_TAG_STRING_OWNED:
        return OwnedValue.make_string(d.get_string(tape_idx))
    if tag == TAPE_TAG_ARRAY:
        var count = d.get_count(tape_idx)
        var child_start = d.get_child_start(tape_idx)
        var items = List[OwnedValue](capacity=count)
        for i in range(count):
            var child = _view_to_owned(doc, child_start + i)
            items.append(child^)
        return OwnedValue.make_array(items^)
    if tag == TAPE_TAG_OBJECT:
        var pair_count = d.get_count(tape_idx)
        var child_start = d.get_child_start(tape_idx)
        var keys = List[String](capacity=pair_count)
        var values = List[OwnedValue](capacity=pair_count)
        for i in range(pair_count):
            var key = d.get_key(child_start + 2 * i)
            keys.append(key^)
            var val = _view_to_owned(doc, child_start + 2 * i + 1)
            values.append(val^)
        return OwnedValue.make_object(keys^, values^)
    raise Error("Unknown tape tag in _view_to_owned: " + String(Int(tag)))


def _parse_owned_value(json_str: String) raises -> OwnedValue:
    """Parse a raw JSON value string into an `OwnedValue` tree.

    Small recursive-descent walker used by mutation paths that
    receive a value as raw JSON. Emits an `OwnedValue` directly,
    matching the shape later turned into a tape-backed `Value` by
    `_serialize_into_value`.
    """
    var bytes = json_str.as_bytes()
    var n = len(bytes)
    var pos = 0

    var result = _parse_owned_at(bytes, pos, n, json_str)
    return result^


def _skip_ws(bytes: Span[UInt8, _], pos: Int, n: Int) -> Int:
    var i = pos
    while i < n and (
        bytes[i] == UInt8(ord(" "))
        or bytes[i] == UInt8(ord("\t"))
        or bytes[i] == UInt8(ord("\n"))
        or bytes[i] == UInt8(ord("\r"))
    ):
        i += 1
    return i


def _parse_owned_at(
    bytes: Span[UInt8, _], start: Int, n: Int, raw_json: String
) raises -> OwnedValue:
    """Parse a single JSON value starting at `start` in `bytes`.

    `raw_json` is passed through only so error messages can include
    context; functionally only `bytes[start:n]` is consumed.
    """
    var i = _skip_ws(bytes, start, n)
    if i >= n:
        raise Error("Unexpected end of JSON")

    var c = bytes[i]

    if c == UInt8(ord("n")):
        return OwnedValue.make_null()
    if c == UInt8(ord("t")):
        return OwnedValue.make_bool(True)
    if c == UInt8(ord("f")):
        return OwnedValue.make_bool(False)
    if c == UInt8(ord('"')):
        var start_idx = i + 1
        var end_idx = start_idx
        var has_escapes = False
        while end_idx < n:
            var b = bytes[end_idx]
            if b == UInt8(ord("\\")):
                has_escapes = True
                end_idx += 2
                continue
            if b == UInt8(ord('"')):
                break
            end_idx += 1
        if not has_escapes:
            return OwnedValue.make_string(
                String(unsafe_from_utf8=bytes[start_idx:end_idx])
            )
        var unescaped = unescape_json_string_span(bytes, start_idx, end_idx)
        return OwnedValue.make_string(String(unsafe_from_utf8=unescaped^))

    if c == UInt8(ord("-")) or (c >= UInt8(ord("0")) and c <= UInt8(ord("9"))):
        var num_start = i
        var is_float = False
        if c == UInt8(ord("-")):
            i += 1
        while i < n:
            var b = bytes[i]
            if (b >= UInt8(ord("0")) and b <= UInt8(ord("9"))) or b == UInt8(
                ord("+")
            ):
                i += 1
            elif (
                b == UInt8(ord("."))
                or b == UInt8(ord("e"))
                or b == UInt8(ord("E"))
            ):
                is_float = True
                i += 1
            elif b == UInt8(ord("-")):
                # Negative exponent.
                i += 1
            else:
                break
        var num_str = String(unsafe_from_utf8=bytes[num_start:i])
        if is_float:
            return OwnedValue.make_float(atof(num_str))
        return OwnedValue.make_int(Int64(atol(num_str)))

    if c == UInt8(ord("[")):
        return _parse_owned_array(bytes, i, n, raw_json)

    if c == UInt8(ord("{")):
        return _parse_owned_object(bytes, i, n, raw_json)

    raise Error("Invalid JSON value")


def _parse_owned_array(
    bytes: Span[UInt8, _], start: Int, n: Int, raw_json: String
) raises -> OwnedValue:
    var i = start + 1  # Skip '['
    var items = List[OwnedValue]()

    i = _skip_ws(bytes, i, n)
    if i < n and bytes[i] == UInt8(ord("]")):
        return OwnedValue.make_array(items^)

    while i < n:
        var elem = _parse_owned_at(bytes, i, n, raw_json)
        items.append(elem^)

        # Advance past the element we just parsed by walking until we hit
        # a top-level comma or the closing bracket.
        i = _skip_to_array_separator(bytes, i, n)

        if i >= n:
            raise Error("Unterminated array")

        if bytes[i] == UInt8(ord(",")):
            i += 1
            i = _skip_ws(bytes, i, n)
            continue
        elif bytes[i] == UInt8(ord("]")):
            return OwnedValue.make_array(items^)
        else:
            raise Error("Expected ',' or ']' in array")

    raise Error("Unterminated array")


def _parse_owned_object(
    bytes: Span[UInt8, _], start: Int, n: Int, raw_json: String
) raises -> OwnedValue:
    var i = start + 1  # Skip '{'
    var keys = List[String]()
    var values = List[OwnedValue]()

    i = _skip_ws(bytes, i, n)
    if i < n and bytes[i] == UInt8(ord("}")):
        return OwnedValue.make_object(keys^, values^)

    while i < n:
        i = _skip_ws(bytes, i, n)
        if i >= n or bytes[i] != UInt8(ord('"')):
            raise Error("Expected string key in object")
        # Parse key.
        var key_start = i + 1
        var key_end = key_start
        while key_end < n:
            var b = bytes[key_end]
            if b == UInt8(ord("\\")):
                key_end += 2
                continue
            if b == UInt8(ord('"')):
                break
            key_end += 1
        var key = String(unsafe_from_utf8=bytes[key_start:key_end])
        keys.append(key^)
        i = key_end + 1
        i = _skip_ws(bytes, i, n)
        if i >= n or bytes[i] != UInt8(ord(":")):
            raise Error("Expected ':' after object key")
        i += 1
        i = _skip_ws(bytes, i, n)

        # Parse value.
        var v = _parse_owned_at(bytes, i, n, raw_json)
        values.append(v^)

        i = _skip_to_object_separator(bytes, i, n)
        if i >= n:
            raise Error("Unterminated object")
        if bytes[i] == UInt8(ord(",")):
            i += 1
            continue
        elif bytes[i] == UInt8(ord("}")):
            return OwnedValue.make_object(keys^, values^)
        else:
            raise Error("Expected ',' or '}' in object")

    raise Error("Unterminated object")


def _skip_to_array_separator(bytes: Span[UInt8, _], start: Int, n: Int) -> Int:
    """Walk past one JSON value, returning the index of the next ',' or ']'.

    Used after parsing an element to advance past its bytes without
    re-parsing. Honors string/escape state so commas inside strings
    are ignored.
    """
    var i = start
    var depth = 0
    var in_string = False
    var escaped = False

    while i < n:
        var c = bytes[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if c == UInt8(ord("\\")) and in_string:
            escaped = True
            i += 1
            continue
        if c == UInt8(ord('"')):
            in_string = not in_string
            i += 1
            continue
        if in_string:
            i += 1
            continue
        if c == UInt8(ord("[")) or c == UInt8(ord("{")):
            depth += 1
        elif c == UInt8(ord("]")) or c == UInt8(ord("}")):
            if depth == 0:
                return i
            depth -= 1
        elif c == UInt8(ord(",")) and depth == 0:
            return i
        i += 1

    return i


def _skip_to_object_separator(bytes: Span[UInt8, _], start: Int, n: Int) -> Int:
    """Same as `_skip_to_array_separator` but stops at ',' or '}'."""
    var i = start
    var depth = 0
    var in_string = False
    var escaped = False

    while i < n:
        var c = bytes[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if c == UInt8(ord("\\")) and in_string:
            escaped = True
            i += 1
            continue
        if c == UInt8(ord('"')):
            in_string = not in_string
            i += 1
            continue
        if in_string:
            i += 1
            continue
        if c == UInt8(ord("[")) or c == UInt8(ord("{")):
            depth += 1
        elif c == UInt8(ord("]")) or c == UInt8(ord("}")):
            if depth == 0:
                return i
            depth -= 1
        elif c == UInt8(ord(",")) and depth == 0:
            return i
        i += 1

    return i


# ---------------------------------------------------------------------------
# OwnedValue serialization
# ---------------------------------------------------------------------------


def _owned_to_json(o: OwnedValue) -> String:
    """Serialize an `OwnedValue` tree to JSON."""
    var w = JsonWriter(capacity=_estimate_owned_bytes(o))
    _write_owned(w, o)
    return w^.finish_string()


def _estimate_owned_bytes(o: OwnedValue) -> Int:
    """Rough output size, to size the writer once.

    A shallow walk: exact for scalars, and for containers it sums child
    estimates rather than guessing a constant, since a hand-built tree
    has no input string to measure against. `ensure` covers any
    shortfall from escaping.
    """
    if o.kind == OWNED_NULL:
        return 4
    if o.kind == OWNED_BOOL:
        return 5
    if o.kind == OWNED_INT:
        return 20
    if o.kind == OWNED_FLOAT:
        return 24
    if o.kind == OWNED_STRING:
        return o.str_val.byte_length() + 2
    if o.kind == OWNED_ARRAY:
        var n = 2
        for i in range(len(o.array_val)):
            n += _estimate_owned_bytes(o.array_val[i]) + 1
        return n
    if o.kind == OWNED_OBJECT:
        var n = 2
        for i in range(len(o.object_keys)):
            n += o.object_keys[i].byte_length() + 4
            n += _estimate_owned_bytes(o.object_values[i]) + 1
        return n
    return 4


def _write_owned(mut w: JsonWriter, o: OwnedValue):
    """Walk an owned tree into `w`, copying each leaf's bytes once."""
    if o.kind == OWNED_NULL:
        w.write_null()
        return
    if o.kind == OWNED_BOOL:
        w.write_bool(o.bool_val)
        return
    if o.kind == OWNED_INT:
        w.write_int(o.int_val)
        return
    if o.kind == OWNED_FLOAT:
        w.write_float(o.float_val)
        return
    if o.kind == OWNED_STRING:
        w.write_string(o.str_val)
        return
    if o.kind == OWNED_ARRAY:
        w.open_container(UInt8(0x5B))
        for i in range(len(o.array_val)):
            w.next_child(i == 0)
            _write_owned(w, o.array_val[i])
        w.close_container(UInt8(0x5D), len(o.array_val) == 0)
        return
    if o.kind == OWNED_OBJECT:
        w.open_container(UInt8(0x7B))
        for i in range(len(o.object_keys)):
            w.next_child(i == 0)
            w.write_string(o.object_keys[i])
            w.colon()
            _write_owned(w, o.object_values[i])
        w.close_container(UInt8(0x7D), len(o.object_keys) == 0)
        return
    w.write_null()


def _escape_json_string(s: String) -> String:
    """Deprecated shim: use `escape_json_string` from `raw_ops`.

    Kept only so existing callers in this module keep working. This used
    to be its own implementation that omitted the U+0000..U+001F range,
    so a control character in a hand-built object serialized to invalid
    JSON while the same value from a tape did not.
    """
    return escape_json_string(s)


# ---------------------------------------------------------------------------
# Value mutation through OwnedValue
# ---------------------------------------------------------------------------


def _materialize_for_write(v: Value) raises -> OwnedValue:
    """Convert a Value into an OwnedValue tree for in-place mutation.

    This is the entry point for the COW path: callers materialize the
    tree, mutate it, then call `_serialize_into_value` to fold the
    result back into a `Value` whose `_raw` reflects the change.
    """
    return _value_to_owned(v)


def _serialize_into_value(o: OwnedValue) -> Value:
    """Serialize an `OwnedValue` into a tape-backed `Value` view.

    Builds a fresh `Document` from the OwnedValue tree (no JSON
    string round-trip, no parsing) and returns a view over it. The
    layout matches what the CPU and FFI parsers emit, so callers
    don't need to care which source produced the value.

    Used by `Value.set` / `Value.append` / `Value.set_at` to fold
    mutations back into a `Value`.
    """
    var doc = _owned_to_tape(o)
    var root_idx = doc.root()
    var arc = ArcPointer[Document](doc^)
    return make_view_value(arc, root_idx)


def _owned_to_tape(o: OwnedValue) -> Document:
    """Build a `Document` whose root entry represents `o`.

    Mirror of the producer protocol used by `_emit_simdjson_value`
    (parser.mojo) and `_emit_value_to_doc` (cpu/stage2.mojo): for
    containers we emit children's headers into a contiguous run
    before writing the parent header, and the root header is
    appended last so `Document.root()` returns its index.
    """
    var doc = Document()
    var root_header = _emit_owned_to_doc(doc, o)
    doc.tape.append(root_header)
    return doc^


def _emit_owned_to_doc(mut doc: Document, o: OwnedValue) -> UInt64:
    """Translate one `OwnedValue` into its tape header. Descendants
    are flushed into `doc.tape` as a side effect; the returned header
    is NOT appended (the caller does that).
    """
    var payload_mask = (UInt64(1) << 60) - 1
    if o.kind == OWNED_NULL:
        return pack_tape_entry(TAPE_TAG_NULL, 0)
    if o.kind == OWNED_BOOL:
        var b: UInt64 = 1 if o.bool_val else 0
        return pack_tape_entry(TAPE_TAG_BOOL, b)
    if o.kind == OWNED_INT:
        return pack_tape_entry(TAPE_TAG_INT, UInt64(o.int_val) & payload_mask)
    if o.kind == OWNED_FLOAT:
        var pool_idx = len(doc.float_pool)
        doc.float_pool.append(o.float_val)
        return pack_tape_entry(TAPE_TAG_FLOAT, UInt64(pool_idx))
    if o.kind == OWNED_STRING:
        var pool_idx = len(doc.string_pool)
        doc.string_pool.append(o.str_val.copy())
        return pack_tape_entry(TAPE_TAG_STRING_OWNED, UInt64(pool_idx))
    if o.kind == OWNED_ARRAY:
        var count = len(o.array_val)
        var headers = List[UInt64](capacity=count)
        for i in range(count):
            headers.append(_emit_owned_to_doc(doc, o.array_val[i]))
        var child_start = len(doc.tape)
        for j in range(len(headers)):
            doc.tape.append(headers[j])
        return pack_tape_entry(
            TAPE_TAG_ARRAY,
            pack_pair(UInt64(count), UInt64(child_start)),
        )
    if o.kind == OWNED_OBJECT:
        var pair_count = len(o.object_keys)
        var headers = List[UInt64](capacity=2 * pair_count)
        for i in range(pair_count):
            var key_pool_idx = len(doc.key_pool)
            doc.key_pool.append(o.object_keys[i].copy())
            headers.append(pack_tape_entry(TAPE_TAG_KEY, UInt64(key_pool_idx)))
            headers.append(_emit_owned_to_doc(doc, o.object_values[i]))
        var child_start = len(doc.tape)
        for j in range(len(headers)):
            doc.tape.append(headers[j])
        return pack_tape_entry(
            TAPE_TAG_OBJECT,
            pack_pair(UInt64(pair_count), UInt64(child_start)),
        )
    return pack_tape_entry(TAPE_TAG_NULL, 0)


# ---------------------------------------------------------------------------
# OwnedValue navigation (used for nested mutation via JSON Pointer)
# ---------------------------------------------------------------------------


def _set_at_pointer(
    mut tree: OwnedValue,
    tokens: List[String],
    idx: Int,
    var new_val: OwnedValue,
) raises:
    """Recursively set the value at the path `tokens[idx:]` in `tree`."""
    if idx == len(tokens):
        # Replace the entire tree -- caller guarantees this branch is
        # only reached at the top level.
        tree = new_val^
        return

    var token = tokens[idx]

    if tree.kind == OWNED_OBJECT:
        var key_pos = -1
        for i in range(len(tree.object_keys)):
            if tree.object_keys[i] == token:
                key_pos = i
                break
        if idx == len(tokens) - 1:
            if key_pos >= 0:
                tree.object_values[key_pos] = new_val^
            else:
                tree.object_keys.append(token)
                tree.object_values.append(new_val^)
        else:
            if key_pos < 0:
                raise Error("Path does not exist in object: /" + token)
            _set_at_pointer(
                tree.object_values[key_pos], tokens, idx + 1, new_val^
            )
        return

    if tree.kind == OWNED_ARRAY:
        var index: Int
        try:
            index = atol(token)
        except:
            raise Error("Array index must be a number: " + token)
        if index < 0 or index >= len(tree.array_val):
            raise Error("Array index out of bounds: " + token)
        if idx == len(tokens) - 1:
            tree.array_val[index] = new_val^
        else:
            _set_at_pointer(tree.array_val[index], tokens, idx + 1, new_val^)
        return

    raise Error(
        "Cannot navigate into primitive value with pointer token: /" + token
    )
