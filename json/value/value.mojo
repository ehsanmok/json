# json - Core Value type.
#
# `Value` carries one of two representations, and which one it holds is
# an implementation detail no caller has to think about:
#
#   read  -- a tape-backed view over a `Document`. `_doc` is `Some`, the
#            data lives in `_doc[].tape[_tape_idx]` plus the side pools,
#            and child views share the `ArcPointer` (a refcount bump, not
#            a copy). This is what every parser produces.
#
#   write -- an owned `OwnedValue` tree in `_owned`, with `_doc` empty.
#            This is what the primitive constructors and the `object()` /
#            `array()` factories produce, and what a tape-backed value
#            converts to the first time it is mutated.
#
# Why two: the tape is a build-once layout. A container's children are a
# contiguous run written *before* the parent header, and the child count
# is packed into the parent entry, so you cannot append one element to an
# existing array in place -- see `document.mojo`. Mutating a tape
# therefore means rebuilding it. Doing that per mutation made growing a
# tree quadratic; keeping the owned tree live across mutations makes it
# O(1) amortized per node, and a conversion happens at most once.
#
# A default `OwnedValue` heap-allocates nothing (see `node.mojo`), so a
# scalar or empty container costs no allocation at all -- where the old
# representation built a whole single-entry `Document` behind an
# `ArcPointer` for every `Value(1)`.

from std.collections import List
from std.memory import ArcPointer

from .raw_ops import _parse_json_pointer, escape_json_string
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
from .owned import (
    _estimate_owned_bytes,
    _owned_to_json,
    _write_owned,
    _set_at_pointer,
    _value_to_owned,
)
from ..writer import JsonWriter
from ..document import (
    Document,
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


struct Null(Writable):
    """Represents JSON null."""

    def __init__(out self):
        pass

    def __str__(self) -> String:
        return "null"

    def write_to[W: Writer](self, mut writer: W):
        writer.write("null")


struct Value(Copyable, Movable, Writable):
    """A JSON value.

    Holds either a tape-backed view over a shared `Document` (what the
    parsers produce) or an owned mutable tree (what the constructors and
    factories produce, and what a parsed value becomes on first
    mutation). See the module header for why both exist; callers use the
    same API either way.
    """

    # `Some` => tape view, and `_tape_idx` is its slot. Empty => `_owned`
    # is authoritative. An `Optional` rather than a sentinel `Document`
    # so the owned representation costs no allocation.
    var _doc: Optional[ArcPointer[Document]]
    var _tape_idx: Int
    var _owned: OwnedValue

    def __init__(out self, doc: ArcPointer[Document], tape_idx: Int):
        """Tape-backed view over `doc[].tape[tape_idx]`.

        The parser entry point: `make_view_value` and `_make_view_child`
        both land here, sharing one `ArcPointer` across every view into
        the same document.
        """
        self._doc = doc
        self._tape_idx = tape_idx
        self._owned = OwnedValue()

    def __init__(out self, var owned: OwnedValue):
        """Owned-tree value. Takes ownership; no document is built."""
        self._doc = None
        self._tape_idx = 0
        self._owned = owned^

    @staticmethod
    def object() -> Value:
        """An empty JSON object, ready to `set` into.

        Allocation-free, and the intended way to build an object from
        scratch. Before this existed the only route to an empty object
        was `loads("{}")` -- a full parser invocation per node.

        Example:
            var o = Value.object()
            o.set("name", Value("Ada"))
        """
        return Value(OwnedValue.make_object(List[String](), List[OwnedValue]()))

    @staticmethod
    def array() -> Value:
        """An empty JSON array, ready to `append` into.

        Allocation-free. See `object()`.

        Example:
            var a = Value.array()
            a.append(Value(1))
        """
        return Value(OwnedValue.make_array(List[OwnedValue]()))

    @always_inline
    def _is_view(self) -> Bool:
        """True when this value reads from a tape."""
        return Bool(self._doc)

    def _to_owned_in_place(mut self) raises:
        """Switch to the owned representation, converting once if needed.

        Called by every mutator. After this the value stays owned, so a
        run of mutations converts at most once rather than per call.
        """
        if self._is_view():
            self._owned = _value_to_owned(self)
            self._doc = None
            self._tape_idx = 0

    def _as_owned(self) raises -> OwnedValue:
        """This value as an owned tree, converting a view if necessary.

        Deep-copies, because the caller only borrowed this value. The
        `var`-taking mutator overloads use `_into_owned` instead and
        avoid the copy entirely.
        """
        if self._is_view():
            return _value_to_owned(self)
        return self._owned.copy()

    def _into_owned(var self) raises -> OwnedValue:
        """Consume this value, yielding its owned tree without copying.

        The move counterpart of `_as_owned`. A tape-backed value still
        has to be materialized; an owned one just hands over its tree.
        """
        if self._is_view():
            return _value_to_owned(self)
        # Swap rather than move the field out: `Value` must stay whole
        # enough to destroy, and Mojo rejects moving a field from the
        # middle of a value. Swapping in a default `OwnedValue` costs
        # nothing -- a default node makes no allocation.
        var out = OwnedValue()
        swap(self._owned, out)
        return out^

    # Scalar constructors. Each builds an owned node, which allocates
    # nothing -- previously each built a single-entry `Document` behind an
    # `ArcPointer`, i.e. one heap allocation per scalar.

    def __init__(out self, null: Null):
        self = Self(OwnedValue.make_null())

    def __init__(out self, none: NoneType):
        self = Self(OwnedValue.make_null())

    def __init__(out self, b: Bool):
        self = Self(OwnedValue.make_bool(b))

    def __init__(out self, i: Int):
        self = Self(OwnedValue.make_int(Int64(i)))

    def __init__(out self, i: Int64):
        self = Self(OwnedValue.make_int(i))

    def __init__(out self, f: Float64):
        self = Self(OwnedValue.make_float(f))

    def __init__(out self, var s: String):
        self = Self(OwnedValue.make_string(s^))

    def copy(self) -> Self:
        """Create a copy of this Value.

        A tape view copies for free -- the document lives behind an
        `ArcPointer`, so this is a refcount bump plus the tape index. An
        owned tree is deep-copied, giving the copy independent value
        semantics.

        Returns:
            A new Value with the same content.
        """
        if self._is_view():
            return Value(self._doc.value().copy(), self._tape_idx)
        return Value(self._owned.copy())

    def clone(self) -> Self:
        """Alias for copy(). Creates a deep copy of this Value.

        Returns:
            A new Value with the same content.
        """
        return self.copy()

    @always_inline
    def _view_tag(self) -> UInt8:
        """Tape tag for the entry this view points at. View mode only."""
        return self._doc.value()[].get_tag(self._tape_idx)

    @always_inline
    def _kind(self) -> Int:
        """The value's kind as an `OWNED_*` constant.

        The single place the two representations are reconciled: every
        predicate below is a comparison against this, so `is_object()`
        means the same thing whether the value came from a parser or was
        built by hand.
        """
        if not self._is_view():
            return self._owned.kind
        var t = self._view_tag()
        if t == TAPE_TAG_NULL:
            return OWNED_NULL
        if t == TAPE_TAG_BOOL:
            return OWNED_BOOL
        if t == TAPE_TAG_INT:
            return OWNED_INT
        if t == TAPE_TAG_FLOAT:
            return OWNED_FLOAT
        if t == TAPE_TAG_STRING or t == TAPE_TAG_STRING_OWNED:
            return OWNED_STRING
        if t == TAPE_TAG_ARRAY:
            return OWNED_ARRAY
        return OWNED_OBJECT

    # Type checking
    def is_null(self) -> Bool:
        return self._kind() == OWNED_NULL

    def is_bool(self) -> Bool:
        return self._kind() == OWNED_BOOL

    def is_int(self) -> Bool:
        return self._kind() == OWNED_INT

    def is_float(self) -> Bool:
        return self._kind() == OWNED_FLOAT

    def is_string(self) -> Bool:
        return self._kind() == OWNED_STRING

    def is_array(self) -> Bool:
        return self._kind() == OWNED_ARRAY

    def is_object(self) -> Bool:
        return self._kind() == OWNED_OBJECT

    def is_number(self) -> Bool:
        var k = self._kind()
        return k == OWNED_INT or k == OWNED_FLOAT

    # Value extraction
    def bool_value(self) -> Bool:
        if self._is_view():
            return self._doc.value()[].get_bool(self._tape_idx)
        return self._owned.bool_val

    def int_value(self) -> Int64:
        if self._is_view():
            return self._doc.value()[].get_int(self._tape_idx)
        return self._owned.int_val

    def float_value(self) -> Float64:
        if self._is_view():
            return self._doc.value()[].get_float(self._tape_idx)
        return self._owned.float_val

    def string_value(self) -> String:
        if self._is_view():
            return self._doc.value()[].get_string(self._tape_idx)
        return self._owned.str_val

    def raw_json(self) -> String:
        if self._is_view():
            return _emit_view_json(self._doc.value(), self._tape_idx)
        return _owned_to_json(self._owned)

    def pretty_json(self, indent: String) -> String:
        """This value as indented JSON.

        One structural walk with an indent-aware writer, rather than
        serializing compactly and re-scanning the text.
        """
        if self._is_view():
            var w = JsonWriter(
                capacity=_estimate_view_bytes(self._doc.value()) * 2,
                indent=indent,
            )
            _write_view(w, self._doc.value(), self._tape_idx)
            return w^.finish_string()
        var ow = JsonWriter(
            capacity=_estimate_owned_bytes(self._owned) * 2, indent=indent
        )
        _write_owned(ow, self._owned)
        return ow^.finish_string()

    def array_count(self) -> Int:
        if self._is_view():
            return self._doc.value()[].get_count(self._tape_idx)
        return len(self._owned.array_val)

    def object_keys(self) -> List[String]:
        if not self._is_view():
            return self._owned.object_keys.copy()
        ref doc = self._doc.value()[]
        var pair_count = doc.get_count(self._tape_idx)
        var child_start = doc.get_child_start(self._tape_idx)
        var keys = List[String](capacity=pair_count)
        for i in range(pair_count):
            keys.append(doc.get_key(child_start + 2 * i))
        return keys^

    def object_count(self) -> Int:
        if self._is_view():
            return self._doc.value()[].get_count(self._tape_idx)
        return len(self._owned.object_keys)

    # Stringable
    def __str__(self) -> String:
        if self._is_view():
            return _emit_view_json(self._doc.value(), self._tape_idx)
        return _owned_to_json(self._owned)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    def __eq__(self, other: Value) -> Bool:
        """Equality compares serialized JSON form.

        Both sides are tape-backed views, so the only stable
        cross-document definition of equality is "same JSON". For
        primitives this still short-circuits through the typed
        accessors before falling back to a full serialize.
        """
        return _view_eq(self, other)

    def __ne__(self, other: Value) -> Bool:
        return not self.__eq__(other)

    def get(self, key: String) raises -> String:
        """Get a field value from a JSON object as a string.

        This is a helper for deserialization. For objects, it parses
        the raw JSON to extract the field value.

        Args:
            key: The field name to extract.

        Returns:
            The raw JSON value as a string.

        Raises:
            Error if not an object or key not found.
        """
        if not self.is_object():
            raise Error("get() can only be called on JSON objects")

        if not self._is_view():
            var pos = self._owned.find_key(key)
            if pos < 0:
                raise Error("Key '" + key + "' not found in JSON object")
            return _owned_to_json(self._owned.object_values[pos])

        ref doc = self._doc.value()[]
        var pair_count = doc.get_count(self._tape_idx)
        var child_start = doc.get_child_start(self._tape_idx)
        for i in range(pair_count):
            if doc.get_key(child_start + 2 * i) == key:
                var v = _make_view_child(
                    self._doc.value(), child_start + 2 * i + 1
                )
                return _view_to_json(v)
        raise Error("Key '" + key + "' not found in JSON object")

    def array_items(self) raises -> List[Value]:
        """Get all items in a JSON array as a list of Values.

        Returns:
            List of Value objects representing array elements.

        Raises:
            Error if not an array.

        Example:
            var data = loads('[1, "hello", true]')
            for item in data.array_items():
                print(item).
        """
        if not self.is_array():
            raise Error("array_items() can only be called on JSON arrays")

        if not self._is_view():
            var owned_items = List[Value](capacity=len(self._owned.array_val))
            for i in range(len(self._owned.array_val)):
                owned_items.append(Value(self._owned.array_val[i].copy()))
            return owned_items^

        ref doc = self._doc.value()[]
        var count = doc.get_count(self._tape_idx)
        var child_start = doc.get_child_start(self._tape_idx)
        var result = List[Value](capacity=count)
        for i in range(count):
            result.append(_make_view_child(self._doc.value(), child_start + i))
        return result^

    def object_items(self) raises -> List[Tuple[String, Value]]:
        """Get all key-value pairs in a JSON object.

        Returns:
            List of (key, value) tuples.

        Raises:
            Error if not an object.

        Example:
            var data = loads('{"a": 1, "b": 2}')
            for pair in data.object_items():
                var key = pair[0]
                var value = pair[1]
                print(key, value).
        """
        if not self.is_object():
            raise Error("object_items() can only be called on JSON objects")

        if not self._is_view():
            var n = len(self._owned.object_keys)
            var owned_pairs = List[Tuple[String, Value]](capacity=n)
            for i in range(n):
                var ok = self._owned.object_keys[i]
                var ov = Value(self._owned.object_values[i].copy())
                owned_pairs.append((ok, ov^))
            return owned_pairs^

        ref doc = self._doc.value()[]
        var pair_count = doc.get_count(self._tape_idx)
        var child_start = doc.get_child_start(self._tape_idx)
        var result = List[Tuple[String, Value]](capacity=pair_count)
        for i in range(pair_count):
            var k = doc.get_key(child_start + 2 * i)
            var v = _make_view_child(self._doc.value(), child_start + 2 * i + 1)
            result.append((k, v^))
        return result^

    def __getitem__(self, index: Int) raises -> Value:
        """Get array element by index.

        Args:
            index: Zero-based array index.

        Returns:
            The Value at the given index.

        Example:
            var arr = loads('[1, 2, 3]')
            print(arr[0])  # Prints 1.
        """
        if not self.is_array():
            raise Error("Index access requires a JSON array")

        if not self._is_view():
            if index < 0 or index >= len(self._owned.array_val):
                raise Error("Array index out of bounds: " + String(index))
            return Value(self._owned.array_val[index].copy())

        var count = self._doc.value()[].get_count(self._tape_idx)
        if index < 0 or index >= count:
            raise Error("Array index out of bounds: " + String(index))
        var child_start = self._doc.value()[].get_child_start(self._tape_idx)
        return _make_view_child(self._doc.value(), child_start + index)

    def __getitem__(self, key: String) raises -> Value:
        """Get object value by key.

        Args:
            key: Object key.

        Returns:
            The Value for the given key.

        Example:
            var obj = loads('{"name": "Alice"}')
            print(obj["name"])  # Prints "Alice".
        """
        if not self.is_object():
            raise Error("Key access requires a JSON object")

        if not self._is_view():
            var pos = self._owned.find_key(key)
            if pos < 0:
                raise Error("Key not found: " + key)
            return Value(self._owned.object_values[pos].copy())

        ref doc = self._doc.value()[]
        var pair_count = doc.get_count(self._tape_idx)
        var child_start = doc.get_child_start(self._tape_idx)
        for i in range(pair_count):
            if doc.get_key(child_start + 2 * i) == key:
                return _make_view_child(
                    self._doc.value(), child_start + 2 * i + 1
                )
        raise Error("Key not found: " + key)

    def set(mut self, key: String, value: Value) raises:
        """Set or update a value in a JSON object.

        Routes through `OwnedValue` so the in-memory tree stays the source of
        truth for the mutation. The view is then re-serialized so subsequent
        reads via `raw_json()` / `__str__()` observe the new state.

        Args:
            key: Object key.
            value: New value to set.

        Example:
            var obj = loads('{"name": "Alice"}')
            obj.set("age", Value(30))
            obj.set("name", Value("Bob"))  # Update existing.
        """
        if not self.is_object():
            raise Error("set() can only be called on JSON objects")

        self._to_owned_in_place()
        self._owned.set_key(key, value._as_owned())

    def set(mut self, var key: String, var value: Value) raises:
        """`set` that consumes its arguments.

        Selected automatically when the value is a temporary -- the
        common `o.set("k", Value(1))` shape -- so the subtree is moved
        in rather than deep-copied. Mojo resolves this against the
        borrowed overload above by argument convention; existing callers
        passing a named variable keep the copying behaviour and their
        value stays usable.
        """
        if not self.is_object():
            raise Error("set() can only be called on JSON objects")

        self._to_owned_in_place()
        self._owned.set_key(key^, value^._into_owned())

    def set(mut self, index: Int, value: Value) raises:
        """Set a value at an array index.

        Args:
            index: Array index (must be valid).
            value: New value to set.

        Example:
            var arr = loads('[1, 2, 3]')
            arr.set(1, Value(20))  # Result is `[1, 20, 3]`.
        """
        if not self.is_array():
            raise Error("set(index) can only be called on JSON arrays")
        if index < 0 or index >= self.array_count():
            raise Error("Array index out of bounds: " + String(index))

        self._to_owned_in_place()
        self._owned.array_val[index] = value._as_owned()

    def set(mut self, index: Int, var value: Value) raises:
        """`set(index, ...)` that consumes its value. See `set(key, ...)`."""
        if not self.is_array():
            raise Error("set(index) can only be called on JSON arrays")
        if index < 0 or index >= self.array_count():
            raise Error("Array index out of bounds: " + String(index))

        self._to_owned_in_place()
        self._owned.array_val[index] = value^._into_owned()

    def append(mut self, value: Value) raises:
        """Append a value to a JSON array.

        Args:
            value: Value to append.

        Example:
            var arr = loads('[1, 2]')
            arr.append(Value(3))  # Result is `[1, 2, 3]`.
        """
        if not self.is_array():
            raise Error("append() can only be called on JSON arrays")

        self._to_owned_in_place()
        self._owned.push(value._as_owned())

    def append(mut self, var value: Value) raises:
        """`append` that consumes its value.

        This is the one that matters for building: appending a freshly
        built element into a growing array no longer deep-copies the
        element's whole subtree.
        """
        if not self.is_array():
            raise Error("append() can only be called on JSON arrays")

        self._to_owned_in_place()
        self._owned.push(value^._into_owned())

    def set_at(mut self, pointer: String, value: Value) raises:
        """Set a nested value via JSON Pointer (RFC 6901).

        Unlike chained `__getitem__`, this propagates the mutation through the
        full parent chain. Intermediate objects/arrays are created or updated
        as required by the pointer; missing scalar parents raise.

        Args:
            pointer: JSON Pointer string (`""` for root, `"/a/b"` for nested).
            value: New value to install at `pointer`.

        Example:
            var doc = loads('{"a":{"b":1}}')
            doc.set_at("/a/b", Value(42))  # Result is `{"a":{"b":42}}`.
        """
        if pointer == "":
            # Whole-document replacement: adopt both halves of the other
            # value's representation, whichever one is live.
            self._doc = value._doc.copy()
            self._tape_idx = value._tape_idx
            self._owned = value._owned.copy()
            return

        var tokens = _parse_json_pointer(pointer)
        self._to_owned_in_place()
        _set_at_pointer(self._owned, tokens, 0, value._as_owned())

    def set_at(mut self, pointer: String, var value: Value) raises:
        """`set_at` that consumes its value. See `set(key, ...)`."""
        if pointer == "":
            self._doc = value._doc.copy()
            self._tape_idx = value._tape_idx
            var taken = OwnedValue()
            swap(value._owned, taken)
            self._owned = taken^
            return
            return

        var tokens = _parse_json_pointer(pointer)
        self._to_owned_in_place()
        _set_at_pointer(self._owned, tokens, 0, value^._into_owned())

    def at(self, pointer: String) raises -> Value:
        """Navigate to a value using JSON Pointer (RFC 6901).

        JSON Pointer syntax:
            "" (empty) = the whole document.
            "/foo" = member "foo" of object.
            "/foo/0" = first element of array "foo".
            "/a~1b" = member "a/b" (/ escaped as ~1).
            "/m~0n" = member "m~n" (~ escaped as ~0).

        Args:
            pointer: JSON Pointer string (e.g., "/users/0/name").

        Returns:
            The Value at the pointer location.

        Raises:
            Error if pointer is invalid or path doesn't exist.

        Example:
            var data = loads('{"users":[{"name":"Alice"}]}')
            var name = data.at("/users/0/name")  # `Value("Alice")`.
        """
        if pointer == "":
            return self.copy()

        if not pointer.startswith("/"):
            raise Error("JSON Pointer must start with '/' or be empty")

        var tokens = _parse_json_pointer(pointer)
        var current = self.copy()
        for i in range(len(tokens)):
            var token = tokens[i]
            if current.is_object():
                current = current[token]
            elif current.is_array():
                var index: Int
                try:
                    index = atol(token)
                except:
                    raise Error("Array index must be a number: " + token)
                if index < 0:
                    raise Error("Array index cannot be negative: " + token)
                current = current[index]
            else:
                raise Error(
                    "Cannot navigate into primitive value with pointer: /"
                    + token
                )
        return current^


# ---------------------------------------------------------------------------
# Value-dependent helpers
# ---------------------------------------------------------------------------


def _value_to_json(v: Value) -> String:
    """Convert a Value to its JSON string representation.

    Retained as a cross-module helper (re-exported from
    `value/__init__.mojo`); it has no in-library callers. The docstring
    used to describe splicing JSON text into a raw object/array string,
    which is not how mutation has worked for some time -- `set` /
    `append` edit the owned tree in place. The full-featured serializer
    is `serialize.dumps`.
    """
    if v.is_null():
        return "null"
    elif v.is_bool():
        return "true" if v.bool_value() else "false"
    elif v.is_int():
        return String(v.int_value())
    elif v.is_float():
        return String(v.float_value())
    elif v.is_string():
        return escape_json_string(v.string_value())
    elif v.is_array() or v.is_object():
        return v.raw_json()
    return "null"


# ---------------------------------------------------------------------------
# Tape-backed view helpers
# ---------------------------------------------------------------------------


def make_view_value(doc: ArcPointer[Document], tape_idx: Int) -> Value:
    """Build a tape-backed `Value` view over `doc[].tape[tape_idx]`.

    Thin wrapper around the `Value(doc, tape_idx)` primary
    constructor. Kept as a free function so callers don't need to
    spell out the type, and so the construction site reads as
    "make a view" rather than "build a Value".
    """
    return Value(doc, tape_idx)


def _make_view_child(doc: ArcPointer[Document], tape_idx: Int) -> Value:
    """Build a child view by sharing the parent's `doc`."""
    return Value(doc, tape_idx)


def _view_to_json(v: Value) -> String:
    """Serialize a value into a JSON string, whichever half is live.

    Used by `Value.raw_json()` / `Value.__str__()` / equality
    comparison. A tape view walks the tape recursively; an owned tree
    serializes directly, without building a tape first. Non-raising: on
    a corrupt tape the function returns a sentinel "<bad-tape>" string,
    which is a strictly better failure mode than panicking inside
    `__str__`.
    """
    if v._is_view():
        return _emit_view_json(v._doc.value(), v._tape_idx)
    return _owned_to_json(v._owned)


def _emit_view_json(doc: ArcPointer[Document], tape_idx: Int) -> String:
    """Serialize a tape entry to JSON.

    Sizes a writer once, then walks the tape into it.
    """
    var w = JsonWriter(capacity=_estimate_view_bytes(doc))
    _write_view(w, doc, tape_idx)
    return w^.finish_string()


def _estimate_view_bytes(doc: ArcPointer[Document]) -> Int:
    """Cheap output-size estimate, to size the writer up front.

    Not exact -- escaping expands and the tape may be a subtree of a
    larger document -- so `ensure` still covers the shortfall. The point
    is only that the common case allocates once. Using the parsed input
    length is O(1) and, for a whole document, very close.
    """
    var input_len = doc[].input.byte_length()
    var est = input_len + 16 if input_len > 0 else doc[].size() * 12 + 16
    return est if est > 32 else 32


def _write_view(mut w: JsonWriter, doc: ArcPointer[Document], tape_idx: Int):
    """Walk a tape entry into `w`.

    Recurses structurally but writes into one buffer, so a leaf's bytes
    are copied exactly once. The previous version returned a fresh
    `String` per node and concatenated it into its parent, which copied
    every leaf's bytes once per ancestor level.

    Non-raising, like the function it replaced: an unrecognised tag
    emits a marker rather than panicking inside `__str__`.
    """
    ref d = doc[]
    var tag = d.get_tag(tape_idx)
    if tag == TAPE_TAG_NULL:
        w.write_null()
        return
    if tag == TAPE_TAG_BOOL:
        w.write_bool(d.get_bool(tape_idx))
        return
    if tag == TAPE_TAG_INT:
        w.write_int(d.get_int(tape_idx))
        return
    if tag == TAPE_TAG_FLOAT:
        w.write_float(d.get_float(tape_idx))
        return
    if tag == TAPE_TAG_STRING or tag == TAPE_TAG_STRING_OWNED:
        w.write_string(d.get_string(tape_idx))
        return
    if tag == TAPE_TAG_ARRAY:
        var count = d.get_count(tape_idx)
        var child_start = d.get_child_start(tape_idx)
        w.open_container(UInt8(0x5B))
        for i in range(count):
            w.next_child(i == 0)
            _write_view(w, doc, child_start + i)
        w.close_container(UInt8(0x5D), count == 0)
        return
    if tag == TAPE_TAG_OBJECT:
        var pair_count = d.get_count(tape_idx)
        var child_start = d.get_child_start(tape_idx)
        w.open_container(UInt8(0x7B))
        for i in range(pair_count):
            w.next_child(i == 0)
            w.write_string(d.get_key(child_start + 2 * i))
            w.colon()
            _write_view(w, doc, child_start + 2 * i + 1)
        w.close_container(UInt8(0x7D), pair_count == 0)
        return
    w.write_literal("<bad-tape>")


def _escape_json_string(s: String) -> String:
    """Deprecated shim: use `escape_json_string` from `raw_ops`."""
    return escape_json_string(s)


def _view_eq(a: Value, b: Value) -> Bool:
    """Equality comparison between two tape-backed views.

    Compares by serialized JSON form for containers; primitives still
    short-circuit through the typed accessors so most comparisons stay
    cheap.
    """
    var sa = a.raw_json() if (a.is_array() or a.is_object()) else String()
    var sb = b.raw_json() if (b.is_array() or b.is_object()) else String()
    if a.is_array() or a.is_object() or b.is_array() or b.is_object():
        if a.is_array() != b.is_array():
            return False
        if a.is_object() != b.is_object():
            return False
        return sa == sb
    if a.is_null() != b.is_null():
        return False
    if a.is_null():
        return True
    if a.is_bool() != b.is_bool():
        return False
    if a.is_bool():
        return a.bool_value() == b.bool_value()
    if a.is_int() != b.is_int():
        return False
    if a.is_int():
        return a.int_value() == b.int_value()
    if a.is_float() != b.is_float():
        return False
    if a.is_float():
        return a.float_value() == b.float_value()
    if a.is_string() != b.is_string():
        return False
    if a.is_string():
        return a.string_value() == b.string_value()
    return False
