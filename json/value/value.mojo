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

from std.collections import Dict, List
from std.hashlib import Hasher
from std.memory import ArcPointer, bitcast

from ..pointer import array_index
from .raw_ops import _parse_json_pointer, escape_json_string
from .node import (
    OwnedValue,
    OWNED_NULL,
    OWNED_BOOL,
    OWNED_INT,
    OWNED_UINT,
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
    TAPE_TAG_INT_POOL,
    TAPE_TAG_UINT,
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


struct Value(
    Boolable, Copyable, Equatable, Hashable, Movable, SizedRaising, Writable
):
    """A JSON value.

    Holds either a tape-backed view over a shared `Document` (what the
    parsers produce) or an owned mutable tree (what the constructors and
    factories produce, and what a parsed value becomes on first
    mutation). See the module header for why both exist; callers use the
    same API either way.

    The type behaves the way a caller who has met a JSON value in any
    other language expects it to: `len` counts elements or members,
    `in` tests membership, `==` is a structural deep comparison, a
    value may key a `Dict`, an array iterates, and an object hands back
    `items()` / `keys()` / `values()`. `Iterable` conformance is the one
    thing missing: the trait requires an iterator alias parameterized
    over the iterable's origin in a shape this compiler cannot spell for
    a type whose iterator must also carry a mode, so `__iter__` is
    defined directly instead. A `for` loop over a `Value` works either
    way, because the loop resolves `__iter__` rather than the trait.
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

    def __init__(out self, u: UInt64):
        """A JSON integer whose magnitude may exceed `Int64.MAX`.

        The unsigned range was readable through `is_uint` / `uint_value`
        from the day the parser learned to keep it, but no constructor
        reached it, so a caller could observe a big integer coming out
        of a document and had no way to put an equal one back in.
        """
        self = Self(OwnedValue.make_uint(u))

    def __init__(out self, i: Int8):
        """A JSON integer widened from a narrower signed type.

        JSON has one number type, so a caller holding an `Int8` from
        elsewhere in a program should not have to widen it by hand
        before it can become a `Value`.
        """
        self = Self(OwnedValue.make_int(Int64(i)))

    def __init__(out self, i: Int16):
        """A JSON integer widened from `Int16`. See `Value(Int8)`."""
        self = Self(OwnedValue.make_int(Int64(i)))

    def __init__(out self, i: Int32):
        """A JSON integer widened from `Int32`. See `Value(Int8)`."""
        self = Self(OwnedValue.make_int(Int64(i)))

    def __init__(out self, u: UInt8):
        """A JSON integer widened from `UInt8`. See `Value(Int8)`."""
        self = Self(OwnedValue.make_int(Int64(u)))

    def __init__(out self, u: UInt16):
        """A JSON integer widened from `UInt16`. See `Value(Int8)`."""
        self = Self(OwnedValue.make_int(Int64(u)))

    def __init__(out self, u: UInt32):
        """A JSON integer widened from `UInt32`. See `Value(Int8)`.

        Widened into the signed range rather than the unsigned one,
        because every `UInt32` fits `Int64` exactly and a value that
        answers `is_int` is the one nearly every caller reaches for.
        """
        self = Self(OwnedValue.make_int(Int64(u)))

    def __init__(out self, f: Float32):
        """A JSON number widened from `Float32`.

        The widening is the plain one, so a `Float32` that cannot name
        its decimal literal exactly keeps the error it already had
        rather than gaining a different one.
        """
        self = Self(OwnedValue.make_float(Float64(f)))

    def __init__(out self, var items: List[Value]) raises:
        """A JSON array holding `items`, in order.

        Building `{"a":[1,2,3]}` used to take a statement per element
        plus one for the array and one for the object. This takes the
        list a caller already has and moves each element's subtree in,
        so nothing is deep-copied on the way.

        Example:
            var a = Value([Value(1), Value(2), Value(3)]).
        """
        var nodes = List[OwnedValue](capacity=len(items))
        for i in range(len(items)):
            var element = items[i].copy()
            nodes.append(element^._into_owned())
        self = Self(OwnedValue.make_array(nodes^))

    def __init__(out self, var members: Dict[String, Value]) raises:
        """A JSON object holding `members`.

        Member order follows the `Dict`'s own iteration order, which it
        does not promise to be insertion order. Build with
        `Value.object()` plus `set` when the emitted order matters, or
        serialize with `sort_keys=True`.

        Example:
            var d = Dict[String, Value]()
            d["a"] = Value(1)
            var o = Value(d^).
        """
        var keys = List[String](capacity=len(members))
        var values = List[OwnedValue](capacity=len(members))
        for entry in members.items():
            keys.append(entry.key)
            var member = entry.value.copy()
            values.append(member^._into_owned())
        self = Self(OwnedValue.make_object(keys^, values^))

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
        return _tag_kind(self._view_tag())

    def type_name(self) -> String:
        """The JSON type of this value, spelled as JSON spells it.

        Exists so the raising accessors can say what they found rather
        than only what they wanted, which is the difference between an
        error a caller can act on and one that sends them back to the
        document with a print statement.
        """
        var k = self._kind()
        if k == OWNED_NULL:
            return "null"
        if k == OWNED_BOOL:
            return "boolean"
        if k == OWNED_INT or k == OWNED_UINT:
            return "integer"
        if k == OWNED_FLOAT:
            return "number"
        if k == OWNED_STRING:
            return "string"
        if k == OWNED_ARRAY:
            return "array"
        return "object"

    # Type checking
    def is_null(self) -> Bool:
        return self._kind() == OWNED_NULL

    def is_bool(self) -> Bool:
        return self._kind() == OWNED_BOOL

    def is_int(self) -> Bool:
        """A signed integer that fits `Int64`.

        A magnitude above `Int64.MAX` answers `is_uint` instead, so
        that a caller reaching for `int_value` cannot be handed a
        wrapped negative number by accident.
        """
        return self._kind() == OWNED_INT

    def is_uint(self) -> Bool:
        """An integer above `Int64.MAX`, readable through `uint_value`."""
        return self._kind() == OWNED_UINT

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
        return k == OWNED_INT or k == OWNED_UINT or k == OWNED_FLOAT

    # Value extraction
    def bool_value(self) -> Bool:
        if self._is_view():
            return self._doc.value()[].get_bool(self._tape_idx)
        return self._owned.bool_val

    def int_value(self) -> Int64:
        if self._is_view():
            return self._doc.value()[].get_int(self._tape_idx)
        return self._owned.int_val

    def uint_value(self) -> UInt64:
        """The value as an unsigned magnitude.

        Accepts any integer entry, so a caller that has checked
        `is_number` and wants the widest reading does not have to test
        the tag first.
        """
        if self._is_view():
            return self._doc.value()[].get_uint(self._tape_idx)
        return bitcast[DType.uint64](self._owned.int_val)

    def float_value(self) -> Float64:
        if self._is_view():
            return self._doc.value()[].get_float(self._tape_idx)
        return self._owned.float_val

    def string_value(self) -> String:
        if self._is_view():
            return self._doc.value()[].get_string(self._tape_idx)
        return self._owned.str_val

    # Raising accessors.
    #
    # `int_value()` and friends above read the slot they are told to
    # read without checking the tag, so `int_value()` on a string
    # answers 0 and a caller who guessed the shape of a document wrong
    # gets a plausible number instead of a complaint. Those stay as
    # they are, because a caller who has already tested `is_int()` should
    # not pay for a second test and should not have to write `try`.
    # The `as_*` family below is for everyone else: it checks the tag
    # first and raises an error naming the type actually present.

    def as_bool(self) raises -> Bool:
        """This value as a boolean, or an error naming what it is.

        Returns:
            The boolean this value holds.

        Raises:
            Error: If this value is not a JSON boolean.
        """
        if not self.is_bool():
            raise Error("expected a boolean, found " + self.type_name())
        return self.bool_value()

    def as_int(self) raises -> Int64:
        """This value as a signed integer, or an error naming what it is.

        An integer too large for `Int64` answers `is_uint` rather than
        `is_int`, so it is rejected here instead of being handed back
        wrapped into a negative number. Reach for `as_uint` for those.

        Returns:
            The integer this value holds.

        Raises:
            Error: If this value is not a signed JSON integer.
        """
        if not self.is_int():
            raise Error("expected an integer, found " + self.type_name())
        return self.int_value()

    def as_uint(self) raises -> UInt64:
        """This value as an unsigned integer, or an error naming what it is.

        Accepts either integer tag, because a caller asking for the
        unsigned reading of `1` means the same thing whichever range
        the parser happened to store it in. A negative integer is
        rejected rather than wrapped.

        Returns:
            The integer this value holds, as an unsigned magnitude.

        Raises:
            Error: If this value is not a JSON integer, or is negative.
        """
        if self.is_uint():
            return self.uint_value()
        if not self.is_int():
            raise Error("expected an integer, found " + self.type_name())
        var signed = self.int_value()
        if signed < 0:
            raise Error(
                "expected an unsigned integer, found the negative value "
                + String(signed)
            )
        return UInt64(signed)

    def as_float(self) raises -> Float64:
        """This value as a float, or an error naming what it is.

        Accepts any JSON number, not only one that was spelled with a
        decimal point. JSON has a single number type, so whether `1` or
        `1.0` reached the parser is a fact about the text rather than
        about the value, and a caller asking for a float would
        otherwise have to test three tags before reading a field that
        is a number in every document they will ever see.

        Returns:
            The number this value holds, widened to `Float64`.

        Raises:
            Error: If this value is not a JSON number.
        """
        if self.is_float():
            return self.float_value()
        if self.is_int():
            return Float64(self.int_value())
        if self.is_uint():
            return Float64(self.uint_value())
        raise Error("expected a number, found " + self.type_name())

    def as_string(self) raises -> String:
        """This value as a string, or an error naming what it is.

        Does not stringify a number or a boolean. A caller who wants
        the JSON text of any value already has `to_json()`, and
        silently accepting `42` here is what turns a mistyped field
        into a bug two layers away.

        Returns:
            The string this value holds, unescaped.

        Raises:
            Error: If this value is not a JSON string.
        """
        if not self.is_string():
            raise Error("expected a string, found " + self.type_name())
        return self.string_value()

    # Non-raising fallbacks, for the third case: a caller who neither
    # wants to test the tag nor to handle an error, and has a sensible
    # answer ready for a field that is absent or the wrong shape.

    def bool_or(self, default: Bool) -> Bool:
        """The boolean this value holds, or `default` if it holds none."""
        if not self.is_bool():
            return default
        return self.bool_value()

    def int_or(self, default: Int64) -> Int64:
        """The signed integer this value holds, or `default` otherwise."""
        if not self.is_int():
            return default
        return self.int_value()

    def uint_or(self, default: UInt64) -> UInt64:
        """The unsigned integer this value holds, or `default` otherwise.

        Matches `as_uint`: either integer tag is accepted, and a
        negative integer falls back rather than wrapping.
        """
        if self.is_uint():
            return self.uint_value()
        if not self.is_int():
            return default
        var signed = self.int_value()
        if signed < 0:
            return default
        return UInt64(signed)

    def float_or(self, default: Float64) -> Float64:
        """The number this value holds as a float, or `default` otherwise.

        Matches `as_float`: any JSON number is accepted.
        """
        if self.is_float():
            return self.float_value()
        if self.is_int():
            return Float64(self.int_value())
        if self.is_uint():
            return Float64(self.uint_value())
        return default

    def string_or(self, default: String) -> String:
        """The string this value holds, or `default` if it holds none."""
        if not self.is_string():
            return default
        return self.string_value()

    def raw_json(self) -> String:
        if self._is_view():
            return _emit_view_json(self._doc.value(), self._tape_idx)
        return _owned_to_json(self._owned)

    def pretty_json(self, indent: String) -> String:
        """This value as indented JSON.

        One structural walk with an indent-aware writer, rather than
        serializing compactly and re-scanning the text.
        """
        return self.to_json(indent=indent)

    def to_json(
        self,
        *,
        indent: String = String(),
        ascii_only: Bool = False,
        escape_solidus: Bool = False,
        sort_keys: Bool = False,
    ) -> String:
        """This value as JSON, with serializer options applied.

        Args:
            indent: One level of indentation. Empty means compact.
            ascii_only: Escape every code point above U+007F.
            escape_solidus: Escape `/` as `\\/`.
            sort_keys: Emit object members ordered by key.

        See `_to_json`. The two escaping options become parameters
        here so the walk that does not use them is the walk that ran
        before they existed."""
        if ascii_only:
            if escape_solidus:
                return self._to_json[True, True](indent, sort_keys)
            return self._to_json[True, False](indent, sort_keys)
        if escape_solidus:
            return self._to_json[False, True](indent, sort_keys)
        return self._to_json[False, False](indent, sort_keys)

    def _to_json[
        ascii_only: Bool, escape_solidus: Bool
    ](self, indent: String, sort_keys: Bool) -> String:
        """This value as JSON, with the writer's options applied.

        Every option is honoured during the structural walk. They used
        to be applied by re-scanning the finished text, which cost a
        second pass and, for the two escaping options, rebuilt the
        string one code point at a time through `chr`, turning every
        byte above 0x7F into a different character.

        Parameters:
            ascii_only: Escape every code point above U+007F.
            escape_solidus: Escape `/` as `\\/`.

        Args:
            indent: One level of indentation. Empty means compact.
            sort_keys: Emit object members ordered by key.
        """
        if self._is_view():
            var w = JsonWriter(
                capacity=_estimate_view_bytes(self._doc.value()) * 2,
                indent=indent,
            )
            _write_view[ascii_only, escape_solidus](
                w, self._doc.value(), self._tape_idx, sort_keys
            )
            return w^.finish_string()
        var ow = JsonWriter(
            capacity=_estimate_owned_bytes(self._owned) * 2, indent=indent
        )
        _write_owned[ascii_only, escape_solidus](ow, self._owned, sort_keys)
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
        """Structural deep equality, as JSON defines sameness.

        Two values are equal when they are the same JSON, not when
        they serialize to the same text. That distinction is the whole
        point: an object's members carry no order, so `{"a":1,"b":2}`
        and `{"b":2,"a":1}` are one value spelled two ways, and a JSON
        number has no type, so `1` and `1.0` are one number spelled two
        ways. Comparing text answered no to both, which made `==`
        unusable for exactly the job callers reached for it to do.

        Numbers are compared in the widest domain that holds them
        exactly rather than by widening everything to `Float64`, so two
        integers beyond the range a `Float64` can name apart stay
        apart. A float equals an integer only when it is integral and
        converts back to that integer exactly, which is also what keeps
        `__hash__` in step with this.

        Walks the two representations directly instead of materializing
        either side, so comparing two parsed documents allocates
        nothing beyond the strings it has to read out of the pools.
        """
        if self._is_view():
            if other._is_view():
                return _view_view_eq(
                    self._doc.value(),
                    self._tape_idx,
                    other._doc.value(),
                    other._tape_idx,
                )
            return _view_owned_eq(
                self._doc.value(), self._tape_idx, other._owned
            )
        if other._is_view():
            return _view_owned_eq(
                other._doc.value(), other._tape_idx, self._owned
            )
        return _owned_eq(self._owned, other._owned)

    def __ne__(self, other: Value) -> Bool:
        return not self.__eq__(other)

    def __hash__[H: Hasher](self, mut hasher: H):
        """Feed this value's digest to `hasher`.

        The `Hashable` requirement, and the reason a `Value` may key a
        `Dict`. The digest itself is `hash_u64`, which is where the
        agreement with `__eq__` is worked out; this hands the finished
        number to whichever hasher the collection chose.

        Parameters:
            H: The hasher type the collection is using.

        Args:
            hasher: The hasher to update.
        """
        hasher.update(self.hash_u64())

    def hash_u64(self) -> UInt64:
        """A hash that agrees with `__eq__`, so a `Value` can key a `Dict`.

        Agreement is the whole constraint, and it is what dictates the
        two unobvious choices here. An object's members carry no order,
        so its members' hashes are combined with an operation that does
        not care about order; anything order-sensitive would hash
        `{"a":1,"b":2}` and `{"b":2,"a":1}` into different buckets after
        `__eq__` has just called them equal. And a number is hashed by
        its value rather than by its tag or its bits, so `1`, `1.0` and
        the same magnitude read out of the unsigned range all land
        together.

        Hashing a container walks it, so it costs what the subtree
        costs. Key a `Dict` by a scalar where you can.
        """
        if self._is_view():
            return _hash_view(self._doc.value(), self._tape_idx)
        return _hash_owned(self._owned)

    def __len__(self) raises -> Int:
        """The number of things this value contains.

        An array answers its element count and an object its member
        count, which is what `len` means everywhere a JSON value is
        used. A string answers its length in bytes, matching
        `String.byte_length()` rather than a count of code points,
        because that is the number the rest of this library measures a
        string in.

        A null, a boolean and a number contain nothing, and they raise
        rather than answering 0. Answering 0 would make `len(v) == 0`
        mean two different things, and a caller reaching for `len` on a
        number has usually reached for the wrong field. Use
        `is_array()` / `is_object()` first if the shape is genuinely
        not known.

        Returns:
            Element count, member count, or the string's byte length.

        Raises:
            Error: If this value is a null, a boolean or a number.
        """
        var k = self._kind()
        if k == OWNED_ARRAY:
            return self.array_count()
        if k == OWNED_OBJECT:
            return self.object_count()
        if k == OWNED_STRING:
            return self.string_value().byte_length()
        raise Error("a " + self.type_name() + " has no length")

    def __bool__(self) -> Bool:
        """Whether this value is truthy, the way JSON callers read JSON.

        Follows the rule every dynamic language that speaks JSON
        settled on: null is false, a boolean is itself, a number is
        false only at zero, and a string, array or object is false only
        when empty. That makes `if doc["items"]:` mean "there are
        items", which is the test callers actually want to write.

        Note this is not `is_null()` inverted: a present-but-empty
        array is false here and not null.
        """
        var k = self._kind()
        if k == OWNED_NULL:
            return False
        if k == OWNED_BOOL:
            return self.bool_value()
        if k == OWNED_INT:
            return self.int_value() != 0
        if k == OWNED_UINT:
            return self.uint_value() != 0
        if k == OWNED_FLOAT:
            return self.float_value() != 0.0
        if k == OWNED_STRING:
            return self.string_value().byte_length() != 0
        if k == OWNED_ARRAY:
            return self.array_count() != 0
        return self.object_count() != 0

    def __contains__(self, key: String) -> Bool:
        """Whether an object has a member named `key`.

        For an array this asks whether some element is that string, so
        `"a" in loads('["a"]')` is true; the two readings cannot
        collide, because an array has no members and an object's
        members are named by strings rather than held as them.
        Anything else contains nothing and answers false rather than
        raising, matching what `in` does in the languages this spelling
        comes from.
        """
        var k = self._kind()
        if k == OWNED_OBJECT:
            return self._find_member(key) >= 0
        if k == OWNED_ARRAY:
            return self._array_contains(Value(key))
        return False

    def __contains__(self, needle: Int) -> Bool:
        """Whether an array holds `needle`, so `3 in arr` reads directly.

        Present as its own overload because an integer literal does not
        become a `Value` on its own, and `Value(3) in arr` is a worse
        thing to have to write than the test it performs.
        """
        return self.__contains__(Value(needle))

    def __contains__(self, needle: Value) -> Bool:
        """Whether an array holds an element structurally equal to `needle`.

        Equality is `Value.__eq__`, so member order inside a candidate
        element does not matter and `Value(1)` finds a `1.0`. For an
        object this tests the member values rather than the names,
        since the names are strings and the `String` overload already
        answers those. Anything else answers false.
        """
        var k = self._kind()
        if k == OWNED_ARRAY:
            return self._array_contains(needle)
        if k == OWNED_OBJECT:
            var n = self.object_count()
            for i in range(n):
                if self._member_value_at(i) == needle:
                    return True
            return False
        return False

    def _array_contains(self, needle: Value) -> Bool:
        """Linear search of an array's elements for `needle`."""
        var n = self.array_count()
        for i in range(n):
            if self._element_at(i) == needle:
                return True
        return False

    @always_inline
    def _element_at(self, index: Int) -> Value:
        """Element `index` of an array, without a bounds check.

        Private, and every caller has already compared `index` against
        `array_count()`. Exists so the membership, iteration and
        equality paths can reach a child without going through the
        raising `__getitem__`.
        """
        if self._is_view():
            ref doc = self._doc.value()[]
            return _make_view_child(
                self._doc.value(), doc.get_child_start(self._tape_idx) + index
            )
        return Value(self._owned.array_val[index].copy())

    @always_inline
    def _member_key_at(self, index: Int) -> String:
        """The name of object member `index`, without a bounds check."""
        if self._is_view():
            ref doc = self._doc.value()[]
            return doc.get_key(doc.get_child_start(self._tape_idx) + 2 * index)
        return self._owned.object_keys[index]

    @always_inline
    def _member_value_at(self, index: Int) -> Value:
        """The value of object member `index`, without a bounds check."""
        if self._is_view():
            ref doc = self._doc.value()[]
            return _make_view_child(
                self._doc.value(),
                doc.get_child_start(self._tape_idx) + 2 * index + 1,
            )
        return Value(self._owned.object_values[index].copy())

    def _find_member(self, key: String) -> Int:
        """Position of the member named `key`, or -1.

        A linear scan in both representations, which is the deliberate
        choice documented on `OwnedValue.find_key`.
        """
        if not self._is_view():
            return self._owned.find_key(key)
        ref doc = self._doc.value()[]
        var pair_count = doc.get_count(self._tape_idx)
        var child_start = doc.get_child_start(self._tape_idx)
        for i in range(pair_count):
            if doc.get_key(child_start + 2 * i) == key:
                return i
        return -1

    def get(self, key: String) raises -> Optional[Value]:
        """The member named `key`, or `None` when there is no such member.

        BREAKING CHANGE in 0.4.0. This used to return the member's raw
        JSON *text* as a `String` and raise when the key was absent,
        which is two surprises at once for a name that every language
        with a mapping type gives to the lookup that tolerates a miss.
        The old behaviour is still available, under the name that says
        what it does: `raw_member(key)`.

        A value that is not an object still raises, matching
        `__getitem__`. Only the absent key is answered with `None`,
        because "this document has no such field" is data and "I called
        a member lookup on an array" is a bug.

        Args:
            key: The member name to look for.

        Returns:
            The member's value, or `None` if the object has no such
            member.

        Raises:
            Error: If this value is not a JSON object.

        Example:
            var data = loads('{"name":"Alice"}')
            var name = data.get("name")
            if name:
                print(name.value().string_value()).
        """
        if not self.is_object():
            raise Error("get() can only be called on JSON objects")
        var pos = self._find_member(key)
        if pos < 0:
            return None
        return self._member_value_at(pos)

    def get(self, key: String, default: Value) raises -> Value:
        """The member named `key`, or `default` when there is no such member.

        The two-argument form of `get`, for the common shape where the
        caller has a fallback ready and does not want to unwrap an
        `Optional` to use it.

        Args:
            key: The member name to look for.
            default: The value to return when the member is absent.

        Returns:
            The member's value, or `default`.

        Raises:
            Error: If this value is not a JSON object.

        Example:
            var port = config.get("port", Value(8080)).
        """
        if not self.is_object():
            raise Error("get() can only be called on JSON objects")
        var pos = self._find_member(key)
        if pos < 0:
            return default.copy()
        return self._member_value_at(pos)

    def raw_member(self, key: String) raises -> String:
        """The member named `key`, serialized back to JSON text.

        This is what `get(key)` did before 0.4.0, under a name that
        says so. It is a serialization, not a lookup: reading a member
        this way and parsing the result costs a full round trip, so
        prefer `get` / `__getitem__` and the typed accessors unless the
        JSON text itself is what you want.

        Args:
            key: The member name to extract.

        Returns:
            The member's value as JSON text.

        Raises:
            Error: If this value is not an object, or has no such
                member.
        """
        if not self.is_object():
            raise Error("raw_member() can only be called on JSON objects")
        var pos = self._find_member(key)
        if pos < 0:
            raise Error("Key '" + key + "' not found in JSON object")
        return self._member_value_at(pos).raw_json()

    def array_items(self) raises -> List[Value]:
        """Every element of a JSON array, as a `List`.

        Prefer iterating the value itself. This builds the whole list
        before the loop body runs once, so a loop that stops early has
        already paid for every element it will not look at, and a large
        array costs a second copy of itself. `for item in data:` walks
        the same elements one at a time. Kept because it predates the
        iterator and callers hold its result.

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
        """Every member of a JSON object, as a `List` of pairs.

        Prefer `items()`. This builds the whole list, with a copy of
        every key and every member subtree, before the loop body runs
        once; `items()` yields one pair at a time. Kept because it
        predates the iterator and callers hold its result.

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

        A negative index counts back from the end, so `arr[-1]` is the
        last element, matching `List` and matching what a caller
        arriving from any other language will try first. Before this,
        a negative index raised, which is a poor answer to a question
        that has an obvious one.

        Args:
            index: Array index. Negative counts from the end.

        Returns:
            The Value at the given index.

        Example:
            var arr = loads('[1, 2, 3]')
            print(arr[0])   # Prints 1.
            print(arr[-1])  # Prints 3.
        """
        if not self.is_array():
            raise Error("Index access requires a JSON array")

        if not self._is_view():
            var pos = _normalize_index(index, len(self._owned.array_val))
            return Value(self._owned.array_val[pos].copy())

        var count = self._doc.value()[].get_count(self._tape_idx)
        var pos = _normalize_index(index, count)
        var child_start = self._doc.value()[].get_child_start(self._tape_idx)
        return _make_view_child(self._doc.value(), child_start + pos)

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

        Routes through `OwnedValue` so the in-memory tree stays the
        source of truth for the mutation, and every read API on *this*
        value observes it immediately.

        The mutation is local to the value it is called on. `__getitem__`
        hands back an independent value rather than a handle into its
        parent, so `doc["a"].set("b", v)` edits a child that `doc` no
        longer shares and `doc` is unchanged. To write through a parent
        chain, name the destination with a JSON Pointer:
        `doc.set_at("/a/b", v)`.

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

        A negative index counts back from the end, as it does for
        `__getitem__`, so `arr.set(-1, v)` replaces the last element.

        Args:
            index: Array index. Negative counts from the end.
            value: New value to set.

        Example:
            var arr = loads('[1, 2, 3]')
            arr.set(1, Value(20))  # Result is `[1, 20, 3]`.
        """
        if not self.is_array():
            raise Error("set(index) can only be called on JSON arrays")
        var pos = _normalize_index(index, self.array_count())

        self._to_owned_in_place()
        self._owned.array_val[pos] = value._as_owned()

    def set(mut self, index: Int, var value: Value) raises:
        """`set(index, ...)` that consumes its value. See `set(key, ...)`."""
        if not self.is_array():
            raise Error("set(index) can only be called on JSON arrays")
        var pos = _normalize_index(index, self.array_count())

        self._to_owned_in_place()
        self._owned.array_val[pos] = value^._into_owned()

    def __setitem__(mut self, key: String, var value: Value) raises:
        """`obj["k"] = v`, the subscript spelling of `set(key, value)`.

        Assignment is how a mapping is written to in every language
        this API is read from, and there was no reason for a type whose
        reads are already subscripts to make its writes a method call.
        Consumes the value, so the common `obj["k"] = Value(1)` moves
        the subtree in rather than copying it.

        Args:
            key: Object key.
            value: New value to install.

        Raises:
            Error: If this value is not a JSON object.
        """
        self.set(key, value^)

    def __setitem__(mut self, index: Int, var value: Value) raises:
        """`arr[0] = v`, the subscript spelling of `set(index, value)`.

        A negative index counts back from the end, matching
        `__getitem__`. Only replaces an existing element; use `append`
        to grow the array.

        Args:
            index: Array index. Negative counts from the end.
            value: New value to install.

        Raises:
            Error: If this value is not an array, or `index` names no
                element.
        """
        self.set(index, value^)

    def remove(mut self, key: String) raises:
        """Remove the member named `key` from an object.

        The counterpart to `set`, which an object had no way to undo:
        removing a member meant rebuilding the object member by member,
        which is what `json/patch.mojo` had to do.

        Args:
            key: The member name to remove.

        Raises:
            Error: If this value is not an object, or has no such
                member. Use `pop(key, default)` shaped code, or test
                with `key in obj`, when absence is expected.
        """
        if not self.is_object():
            raise Error("remove(key) can only be called on JSON objects")
        self._to_owned_in_place()
        _ = self._owned.take_key(key)

    def remove(mut self, index: Int) raises:
        """Remove element `index` from an array, shifting the rest down.

        A negative index counts back from the end, matching
        `__getitem__`.

        Args:
            index: Array index. Negative counts from the end.

        Raises:
            Error: If this value is not an array, or `index` names no
                element.
        """
        if not self.is_array():
            raise Error("remove(index) can only be called on JSON arrays")
        var pos = _normalize_index(index, self.array_count())
        self._to_owned_in_place()
        _ = self._owned.take_index(pos)

    def pop(mut self, key: String) raises -> Value:
        """Remove the member named `key` and return its value.

        `remove` when the removed value is not wanted, `pop` when it
        is. The value is moved out of the tree rather than copied, so
        popping a large subtree costs no more than popping a scalar.

        Args:
            key: The member name to remove.

        Returns:
            The value that was stored under `key`.

        Raises:
            Error: If this value is not an object, or has no such
                member.
        """
        if not self.is_object():
            raise Error("pop(key) can only be called on JSON objects")
        self._to_owned_in_place()
        return Value(self._owned.take_key(key))

    def pop(mut self, index: Int) raises -> Value:
        """Remove element `index` from an array and return it.

        A negative index counts back from the end, so `arr.pop(-1)`
        takes the last element.

        Args:
            index: Array index. Negative counts from the end.

        Returns:
            The element that was stored at `index`.

        Raises:
            Error: If this value is not an array, or `index` names no
                element.
        """
        if not self.is_array():
            raise Error("pop(index) can only be called on JSON arrays")
        var pos = _normalize_index(index, self.array_count())
        self._to_owned_in_place()
        return Value(self._owned.take_index(pos))

    def pop(mut self) raises -> Value:
        """Remove the last element of an array and return it.

        The no-argument form, so using an array as a stack does not
        need the length computed at the call site.

        Returns:
            The element that was last in the array.

        Raises:
            Error: If this value is not an array, or the array is
                empty.
        """
        if not self.is_array():
            raise Error("pop() can only be called on JSON arrays")
        var count = self.array_count()
        if count == 0:
            raise Error("pop() from an empty array")
        self._to_owned_in_place()
        return Value(self._owned.take_index(count - 1))

    def __iter__(ref self) raises -> ValueArrayIter[origin_of(self)]:
        """Iterate an array's elements, one at a time.

        Lazy, unlike `array_items()`: nothing is built ahead of the
        loop, so a loop that stops at the first match pays for what it
        looked at and no more. The iterator borrows this value, so the
        value has to outlive the loop, which for the usual
        `for item in doc["items"]:` it does.

        Returns:
            An iterator over the array's elements.

        Raises:
            Error: If this value is not a JSON array. Iterating an
                object is ambiguous enough that it is spelled out:
                `items()`, `keys()` or `values()`.

        Example:
            var data = loads('[1, 2, 3]')
            for item in data:
                print(item.int_value()).
        """
        if not self.is_array():
            raise Error(
                "iteration requires a JSON array, found " + self.type_name()
            )
        return ValueArrayIter(self)

    def items(ref self) raises -> ValueItemsIter[origin_of(self)]:
        """Iterate an object's members as `(key, value)` pairs.

        The lazy counterpart to `object_items()`, which copies every
        key and every member subtree before the loop starts. Members
        come back in the order the object holds them, which for a
        parsed document is document order.

        Returns:
            An iterator over `(key, value)` pairs.

        Raises:
            Error: If this value is not a JSON object.

        Example:
            var data = loads('{"a": 1, "b": 2}')
            for pair in data.items():
                print(pair[0], pair[1]).
        """
        if not self.is_object():
            raise Error(
                "items() requires a JSON object, found " + self.type_name()
            )
        return ValueItemsIter(self)

    def keys(ref self) raises -> ValueKeysIter[origin_of(self)]:
        """Iterate an object's member names.

        The lazy counterpart to `object_keys()`, which materializes a
        `List[String]`. Prefer this when the loop only reads the names.

        Returns:
            An iterator over the object's member names.

        Raises:
            Error: If this value is not a JSON object.
        """
        if not self.is_object():
            raise Error(
                "keys() requires a JSON object, found " + self.type_name()
            )
        return ValueKeysIter(self)

    def values(ref self) raises -> ValueValuesIter[origin_of(self)]:
        """Iterate an object's member values, without their names.

        Returns:
            An iterator over the object's member values.

        Raises:
            Error: If this value is not a JSON object.
        """
        if not self.is_object():
            raise Error(
                "values() requires a JSON object, found " + self.type_name()
            )
        return ValueValuesIter(self)

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
                # RFC 6901 spells an index as `0` or a leading non-zero
                # digit, so `01` and `+1` are malformed rather than out
                # of range, and `-` names a position that no existing
                # element occupies.
                current = current[array_index(token, current.array_count())]
            else:
                raise Error(
                    "Cannot navigate into primitive value with pointer: /"
                    + token
                )
        return current^

    def try_at(self, pointer: String) raises -> Optional[Value]:
        """The value at `pointer`, or `None` if nothing is there.

        The twin of `at`, for the case `at` handles badly: probing a
        document for a field that may or may not be present. `at`
        raises on the first missing segment, so probing meant wrapping
        the call in `try`, which also swallows the errors worth
        hearing about.

        A pointer that is not a well-formed JSON Pointer still raises,
        because that is a mistake in the program rather than a fact
        about the document. Only a segment that names nothing, or one
        that tries to descend into a scalar, answers `None`.

        Args:
            pointer: JSON Pointer string. `""` names the whole value.

        Returns:
            The value at the pointer location, or `None`.

        Raises:
            Error: If `pointer` is not a well-formed JSON Pointer.

        Example:
            var data = loads('{"a":{"b":1}}')
            var hit = data.try_at("/a/b")     # `Some(Value(1))`.
            var miss = data.try_at("/a/zz")   # `None`.
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
                var pos = current._find_member(token)
                if pos < 0:
                    return None
                current = current._member_value_at(pos)
            elif current.is_array():
                var count = current.array_count()
                var index: Int
                try:
                    index = array_index(token, count)
                except:
                    # A malformed or out-of-range index names no
                    # element, which is the same "not there" answer a
                    # missing member gets.
                    return None
                current = current._element_at(index)
            else:
                return None
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


def write_value(mut w: JsonWriter, v: Value):
    """Emit `v` into an existing writer.

    The point is the writer: a caller already building a document can
    splice a `Value` into it without the value first becoming its own
    `String` and then being copied in.
    """
    if v._is_view():
        _write_view(w, v._doc.value(), v._tape_idx)
    else:
        _write_owned(w, v._owned)


def _member_order(d: Document, child_start: Int, pair_count: Int) -> List[Int]:
    """Positions of an object's members ordered by key.

    Only reached when sorting was asked for, because the allocation it
    needs would otherwise cost a heap block per object emitted.
    Ordering by bytes is ordering by code point for UTF-8, and an
    insertion sort keeps equal keys in the order they were parsed, so a
    document carrying a repeated name still round-trips.
    """
    var order = List[Int](capacity=pair_count)
    for i in range(pair_count):
        order.append(i)

    var keys = List[String](capacity=pair_count)
    for i in range(pair_count):
        keys.append(d.get_key(child_start + 2 * i))
    for i in range(1, pair_count):
        var slot = order[i]
        var j = i - 1
        while j >= 0 and keys[order[j]] > keys[slot]:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = slot
    return order^


def _write_view[
    ascii_only: Bool = False, solidus: Bool = False
](
    mut w: JsonWriter,
    doc: ArcPointer[Document],
    tape_idx: Int,
    sort_keys: Bool = False,
):
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
    if tag == TAPE_TAG_INT or tag == TAPE_TAG_INT_POOL:
        w.write_int(d.get_int(tape_idx))
        return
    if tag == TAPE_TAG_UINT:
        w.write_uint(d.get_uint(tape_idx))
        return
    if tag == TAPE_TAG_FLOAT:
        w.write_float(d.get_float(tape_idx))
        return
    if tag == TAPE_TAG_STRING or tag == TAPE_TAG_STRING_OWNED:
        w.write_string[ascii_only, solidus](d.get_string(tape_idx))
        return
    if tag == TAPE_TAG_ARRAY:
        var count = d.get_count(tape_idx)
        var child_start = d.get_child_start(tape_idx)
        w.open_container(UInt8(0x5B))
        for i in range(count):
            w.next_child(i == 0)
            _write_view[ascii_only, solidus](w, doc, child_start + i, sort_keys)
        w.close_container(UInt8(0x5D), count == 0)
        return
    if tag == TAPE_TAG_OBJECT:
        var pair_count = d.get_count(tape_idx)
        var child_start = d.get_child_start(tape_idx)
        w.open_container(UInt8(0x7B))
        if sort_keys and pair_count > 1:
            var order = _member_order(d, child_start, pair_count)
            for i in range(pair_count):
                var slot = order[i]
                w.next_child(i == 0)
                w.write_string[ascii_only, solidus](
                    d.get_key(child_start + 2 * slot)
                )
                w.colon()
                _write_view[ascii_only, solidus](
                    w, doc, child_start + 2 * slot + 1, sort_keys
                )
        else:
            for i in range(pair_count):
                w.next_child(i == 0)
                w.write_string[ascii_only, solidus](
                    d.get_key(child_start + 2 * i)
                )
                w.colon()
                _write_view[ascii_only, solidus](
                    w, doc, child_start + 2 * i + 1, sort_keys
                )
        w.close_container(UInt8(0x7D), pair_count == 0)
        return
    w.write_literal("<bad-tape>")


def _escape_json_string(s: String) -> String:
    """Deprecated shim: use `escape_json_string` from `raw_ops`."""
    return escape_json_string(s)


# ---------------------------------------------------------------------------
# Shared small helpers
# ---------------------------------------------------------------------------


@always_inline
def _tag_kind(tag: UInt8) -> Int:
    """The `OWNED_*` kind a tape tag stands for.

    The one place tape tags are mapped onto the dense kind space, so
    `Value._kind` and the structural walks below cannot drift apart on
    what a tag means. Inlined, because `_kind` sits behind every type
    predicate and therefore behind `__getitem__`.
    """
    if tag == TAPE_TAG_NULL:
        return OWNED_NULL
    if tag == TAPE_TAG_BOOL:
        return OWNED_BOOL
    if tag == TAPE_TAG_INT or tag == TAPE_TAG_INT_POOL:
        return OWNED_INT
    if tag == TAPE_TAG_UINT:
        return OWNED_UINT
    if tag == TAPE_TAG_FLOAT:
        return OWNED_FLOAT
    if tag == TAPE_TAG_STRING or tag == TAPE_TAG_STRING_OWNED:
        return OWNED_STRING
    if tag == TAPE_TAG_ARRAY:
        return OWNED_ARRAY
    return OWNED_OBJECT


@always_inline
def _normalize_index(index: Int, count: Int) raises -> Int:
    """`index` as a position in `0 ..< count`, counting back if negative.

    One definition shared by every array subscript, so `arr[-1]`,
    `arr.set(-1, v)` and `arr.pop(-1)` cannot disagree about which
    element they name. The error reports the index the caller wrote
    rather than the normalized one, which is the number they can find
    in their own source.
    """
    var pos = index + count if index < 0 else index
    if pos < 0 or pos >= count:
        raise Error("Array index out of bounds: " + String(index))
    return pos


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


struct _Num(Copyable, Movable):
    """A JSON number lifted out of whichever representation held it.

    Equality and hashing both have to treat a number as a number
    rather than as a tag, and both have to do it from three different
    walks. Carrying the three readings in one small register-sized
    struct is what lets them share a single definition of "the same
    number" instead of growing three that disagree.
    """

    var kind: Int
    var int_val: Int64
    var uint_val: UInt64
    var float_val: Float64

    def __init__(out self, kind: Int, i: Int64, u: UInt64, f: Float64):
        self.kind = kind
        self.int_val = i
        self.uint_val = u
        self.float_val = f


@always_inline
def _num_int(i: Int64) -> _Num:
    return _Num(OWNED_INT, i, 0, 0.0)


@always_inline
def _num_uint(u: UInt64) -> _Num:
    return _Num(OWNED_UINT, 0, u, 0.0)


@always_inline
def _num_float(f: Float64) -> _Num:
    return _Num(OWNED_FLOAT, 0, 0, f)


def _owned_num(o: OwnedValue) -> _Num:
    """The number an owned node holds."""
    if o.kind == OWNED_INT:
        return _num_int(o.int_val)
    if o.kind == OWNED_UINT:
        return _num_uint(bitcast[DType.uint64](o.int_val))
    return _num_float(o.float_val)


def _view_num(doc: ArcPointer[Document], tape_idx: Int, kind: Int) -> _Num:
    """The number a tape entry holds."""
    if kind == OWNED_INT:
        return _num_int(doc[].get_int(tape_idx))
    if kind == OWNED_UINT:
        return _num_uint(doc[].get_uint(tape_idx))
    return _num_float(doc[].get_float(tape_idx))


def _float_eq_integer(f: Float64, n: _Num) -> Bool:
    """Whether the float `f` is exactly the integer `n`.

    Decided in the integer domain rather than by widening `n` to a
    `Float64`, because widening loses the low bits of any magnitude
    above 2^53 and would then report two different integers as equal
    to one float. Converting the other way is exact when it round
    trips, and the round-trip test also rejects a fractional `f`
    without a separate floor.
    """
    if n.kind == OWNED_UINT:
        if f < 0.0 or f >= 18446744073709551616.0:
            return False
        var u = UInt64(f)
        return Float64(u) == f and u == n.uint_val
    if f < -9223372036854775808.0 or f >= 9223372036854775808.0:
        return False
    var i = Int64(f)
    return Float64(i) == f and i == n.int_val


def _num_eq(a: _Num, b: _Num) -> Bool:
    """Whether two JSON numbers are the same number.

    JSON has one number type, so `1` and `1.0` are the same value even
    though the parser stored them under different tags. Integers are
    compared as integers whatever their sign, so a magnitude that
    needed the unsigned range is never confused with a negative one.
    """
    if a.kind == OWNED_FLOAT and b.kind == OWNED_FLOAT:
        return a.float_val == b.float_val
    if a.kind == OWNED_FLOAT:
        return _float_eq_integer(a.float_val, b)
    if b.kind == OWNED_FLOAT:
        return _float_eq_integer(b.float_val, a)
    if a.kind == OWNED_UINT and b.kind == OWNED_UINT:
        return a.uint_val == b.uint_val
    if a.kind == OWNED_UINT:
        return b.int_val >= 0 and a.uint_val == UInt64(b.int_val)
    if b.kind == OWNED_UINT:
        return a.int_val >= 0 and b.uint_val == UInt64(a.int_val)
    return a.int_val == b.int_val


def _num_hash(n: _Num) -> UInt64:
    """A hash that gives every spelling of one number the same answer.

    Anything integral is hashed by its integer value, whichever tag it
    arrived under, so `1`, `1.0` and an unsigned `1` collide the way
    `_num_eq` says they must. `-0.0` is integral too, so it lands with
    `0`. Only a float that is fractional or outside the integer range
    is hashed by its bits, and `_num_eq` compares those by value as
    well, where equal floats have equal bits.
    """
    if n.kind == OWNED_INT:
        return _mix(2, bitcast[DType.uint64](n.int_val))
    if n.kind == OWNED_UINT:
        if n.uint_val <= UInt64(Int64.MAX):
            return _mix(2, n.uint_val)
        return _mix(3, n.uint_val)
    var f = n.float_val
    if f >= -9223372036854775808.0 and f < 9223372036854775808.0:
        var i = Int64(f)
        if Float64(i) == f:
            return _mix(2, bitcast[DType.uint64](i))
    if f >= 9223372036854775808.0 and f < 18446744073709551616.0:
        var u = UInt64(f)
        if Float64(u) == f:
            return _mix(3, u)
    return _mix(4, bitcast[DType.uint64](f))


# ---------------------------------------------------------------------------
# Structural equality
# ---------------------------------------------------------------------------
#
# Three walks rather than one, because the two representations store a
# container's children in genuinely different places and the mixed pair
# has to read one of each. Converting one side first would be a single
# walk, but it would allocate a copy of a whole subtree to answer a
# question that needs no copy at all.


def _owned_eq(a: OwnedValue, b: OwnedValue) -> Bool:
    """Deep equality between two owned trees."""
    var ka = a.kind
    var kb = b.kind
    if ka == OWNED_NULL or kb == OWNED_NULL:
        return ka == kb
    if ka == OWNED_BOOL or kb == OWNED_BOOL:
        return ka == kb and a.bool_val == b.bool_val
    if ka == OWNED_STRING or kb == OWNED_STRING:
        return ka == kb and a.str_val == b.str_val
    if ka == OWNED_ARRAY or kb == OWNED_ARRAY:
        if ka != kb or len(a.array_val) != len(b.array_val):
            return False
        for i in range(len(a.array_val)):
            if not _owned_eq(a.array_val[i], b.array_val[i]):
                return False
        return True
    if ka == OWNED_OBJECT or kb == OWNED_OBJECT:
        if ka != kb or len(a.object_keys) != len(b.object_keys):
            return False
        for i in range(len(a.object_keys)):
            var pos = b.find_key(a.object_keys[i])
            if pos < 0:
                return False
            if not _owned_eq(a.object_values[i], b.object_values[pos]):
                return False
        return True
    return _num_eq(_owned_num(a), _owned_num(b))


def _view_view_eq(
    da: ArcPointer[Document], ia: Int, db: ArcPointer[Document], ib: Int
) -> Bool:
    """Deep equality between two tape entries, possibly in one document."""
    var ka = _tag_kind(da[].get_tag(ia))
    var kb = _tag_kind(db[].get_tag(ib))
    if ka == OWNED_NULL or kb == OWNED_NULL:
        return ka == kb
    if ka == OWNED_BOOL or kb == OWNED_BOOL:
        return ka == kb and da[].get_bool(ia) == db[].get_bool(ib)
    if ka == OWNED_STRING or kb == OWNED_STRING:
        return ka == kb and da[].get_string(ia) == db[].get_string(ib)
    if ka == OWNED_ARRAY or kb == OWNED_ARRAY:
        if ka != kb:
            return False
        var count = da[].get_count(ia)
        if count != db[].get_count(ib):
            return False
        var start_a = da[].get_child_start(ia)
        var start_b = db[].get_child_start(ib)
        for i in range(count):
            if not _view_view_eq(da, start_a + i, db, start_b + i):
                return False
        return True
    if ka == OWNED_OBJECT or kb == OWNED_OBJECT:
        if ka != kb:
            return False
        var pairs = da[].get_count(ia)
        if pairs != db[].get_count(ib):
            return False
        var start_a = da[].get_child_start(ia)
        var start_b = db[].get_child_start(ib)
        for i in range(pairs):
            var key = da[].get_key(start_a + 2 * i)
            var found = -1
            for j in range(pairs):
                if db[].get_key(start_b + 2 * j) == key:
                    found = j
                    break
            if found < 0:
                return False
            if not _view_view_eq(
                da, start_a + 2 * i + 1, db, start_b + 2 * found + 1
            ):
                return False
        return True
    return _num_eq(_view_num(da, ia, ka), _view_num(db, ib, kb))


def _view_owned_eq(
    doc: ArcPointer[Document], tape_idx: Int, o: OwnedValue
) -> Bool:
    """Deep equality between a tape entry and an owned tree.

    The mixed pair, which is what comparing a parsed document against
    a hand-built expectation reduces to, and therefore what most tests
    in this repository end up calling.
    """
    var ka = _tag_kind(doc[].get_tag(tape_idx))
    var kb = o.kind
    if ka == OWNED_NULL or kb == OWNED_NULL:
        return ka == kb
    if ka == OWNED_BOOL or kb == OWNED_BOOL:
        return ka == kb and doc[].get_bool(tape_idx) == o.bool_val
    if ka == OWNED_STRING or kb == OWNED_STRING:
        return ka == kb and doc[].get_string(tape_idx) == o.str_val
    if ka == OWNED_ARRAY or kb == OWNED_ARRAY:
        if ka != kb:
            return False
        var count = doc[].get_count(tape_idx)
        if count != len(o.array_val):
            return False
        var child_start = doc[].get_child_start(tape_idx)
        for i in range(count):
            if not _view_owned_eq(doc, child_start + i, o.array_val[i]):
                return False
        return True
    if ka == OWNED_OBJECT or kb == OWNED_OBJECT:
        if ka != kb:
            return False
        var pairs = doc[].get_count(tape_idx)
        if pairs != len(o.object_keys):
            return False
        var child_start = doc[].get_child_start(tape_idx)
        for i in range(pairs):
            var key = doc[].get_key(child_start + 2 * i)
            var pos = o.find_key(key)
            if pos < 0:
                return False
            if not _view_owned_eq(
                doc, child_start + 2 * i + 1, o.object_values[pos]
            ):
                return False
        return True
    return _num_eq(_view_num(doc, tape_idx, ka), _owned_num(o))


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


@always_inline
def _mix(tag: UInt64, x: UInt64) -> UInt64:
    """Scramble `x` into a hash, keyed by the kind `tag` it came from.

    A SplitMix64 finalizer. The tag separates the kind spaces, so a
    string that hashes its bytes to some number cannot collide with the
    integer of that value for free; and the avalanche is what lets the
    order-insensitive combination below add member hashes together
    without the sum degenerating on objects whose members are small
    integers.
    """
    var h = x + UInt64(0x9E3779B97F4A7C15) * (tag + 1)
    h ^= h >> 30
    h *= UInt64(0xBF58476D1CE4E5B9)
    h ^= h >> 27
    h *= UInt64(0x94D049BB133111EB)
    h ^= h >> 31
    return h


def _hash_bytes(s: String) -> UInt64:
    """FNV-1a over a string's bytes.

    Over bytes rather than code points, because two strings are equal
    here exactly when their bytes are, and decoding would cost a pass
    to reach the same answer.
    """
    var h = UInt64(0xCBF29CE484222325)
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        h ^= UInt64(Int(bytes[i]))
        h *= UInt64(0x100000001B3)
    return h


def _hash_owned(o: OwnedValue) -> UInt64:
    """Hash an owned tree, agreeing with `_owned_eq`."""
    if o.kind == OWNED_NULL:
        return _mix(0, 0)
    if o.kind == OWNED_BOOL:
        return _mix(1, UInt64(1) if o.bool_val else UInt64(0))
    if o.kind == OWNED_STRING:
        return _mix(5, _hash_bytes(o.str_val))
    if o.kind == OWNED_ARRAY:
        # Order matters for an array and only for an array, so this one
        # chains rather than accumulating commutatively.
        var h = UInt64(6)
        for i in range(len(o.array_val)):
            h = _mix(6, h ^ _hash_owned(o.array_val[i]))
        return h
    if o.kind == OWNED_OBJECT:
        var acc = UInt64(0)
        for i in range(len(o.object_keys)):
            acc += _mix(
                7,
                _hash_bytes(o.object_keys[i]) ^ _hash_owned(o.object_values[i]),
            )
        return _mix(7, acc)
    return _num_hash(_owned_num(o))


def _hash_view(doc: ArcPointer[Document], tape_idx: Int) -> UInt64:
    """Hash a tape entry, agreeing with `_hash_owned` value for value."""
    var kind = _tag_kind(doc[].get_tag(tape_idx))
    if kind == OWNED_NULL:
        return _mix(0, 0)
    if kind == OWNED_BOOL:
        return _mix(1, UInt64(1) if doc[].get_bool(tape_idx) else UInt64(0))
    if kind == OWNED_STRING:
        return _mix(5, _hash_bytes(doc[].get_string(tape_idx)))
    if kind == OWNED_ARRAY:
        var count = doc[].get_count(tape_idx)
        var child_start = doc[].get_child_start(tape_idx)
        var h = UInt64(6)
        for i in range(count):
            h = _mix(6, h ^ _hash_view(doc, child_start + i))
        return h
    if kind == OWNED_OBJECT:
        var pairs = doc[].get_count(tape_idx)
        var child_start = doc[].get_child_start(tape_idx)
        var acc = UInt64(0)
        for i in range(pairs):
            acc += _mix(
                7,
                _hash_bytes(doc[].get_key(child_start + 2 * i))
                ^ _hash_view(doc, child_start + 2 * i + 1),
            )
        return _mix(7, acc)
    return _num_hash(_view_num(doc, tape_idx, kind))


# ---------------------------------------------------------------------------
# Lazy iterators
# ---------------------------------------------------------------------------
#
# Each borrows the value it walks rather than owning a snapshot of it,
# so starting a loop costs nothing and stopping one early pays for
# nothing. That is the whole difference from `array_items()` /
# `object_items()`, which build the entire result before the first
# iteration of the loop body.


struct ValueArrayIter[mut: Bool, //, origin: Origin[mut=mut]](
    Copyable, Iterator, Movable
):
    """Yields the elements of a JSON array, one at a time.

    Built by `Value.__iter__`, so `for item in arr:` reaches it.
    """

    comptime Element = Value

    var _src: Pointer[Value, Self.origin]
    var _index: Int
    var _count: Int

    def __init__(out self, ref[Self.origin] src: Value):
        """Start at the first element of `src`."""
        self._src = Pointer(to=src)
        self._index = 0
        self._count = src.array_count()

    def __iter__(self) -> Self:
        """An iterator is its own iterable.

        Present so `for pair in obj.items():` reaches the loop
        protocol directly, the way a `for` loop over any other
        iterator does.
        """
        return self.copy()

    def __has_next__(self) -> Bool:
        """Whether any element remains."""
        return self._index < self._count

    def __next__(mut self) -> Value:
        """The next element."""
        var item = self._src[]._element_at(self._index)
        self._index += 1
        return item^


struct ValueItemsIter[mut: Bool, //, origin: Origin[mut=mut]](
    Copyable, Iterator, Movable
):
    """Yields an object's members as `(key, value)` pairs.

    Built by `Value.items()`.
    """

    comptime Element = Tuple[String, Value]

    var _src: Pointer[Value, Self.origin]
    var _index: Int
    var _count: Int

    def __init__(out self, ref[Self.origin] src: Value):
        """Start at the first member of `src`."""
        self._src = Pointer(to=src)
        self._index = 0
        self._count = src.object_count()

    def __iter__(self) -> Self:
        """An iterator is its own iterable.

        Present so `for pair in obj.items():` reaches the loop
        protocol directly, the way a `for` loop over any other
        iterator does.
        """
        return self.copy()

    def __has_next__(self) -> Bool:
        """Whether any member remains."""
        return self._index < self._count

    def __next__(mut self) -> Tuple[String, Value]:
        """The next `(key, value)` pair."""
        var key = self._src[]._member_key_at(self._index)
        var value = self._src[]._member_value_at(self._index)
        self._index += 1
        return (key^, value^)


struct ValueKeysIter[mut: Bool, //, origin: Origin[mut=mut]](
    Copyable, Iterator, Movable
):
    """Yields an object's member names. Built by `Value.keys()`."""

    comptime Element = String

    var _src: Pointer[Value, Self.origin]
    var _index: Int
    var _count: Int

    def __init__(out self, ref[Self.origin] src: Value):
        """Start at the first member of `src`."""
        self._src = Pointer(to=src)
        self._index = 0
        self._count = src.object_count()

    def __iter__(self) -> Self:
        """An iterator is its own iterable.

        Present so `for pair in obj.items():` reaches the loop
        protocol directly, the way a `for` loop over any other
        iterator does.
        """
        return self.copy()

    def __has_next__(self) -> Bool:
        """Whether any member remains."""
        return self._index < self._count

    def __next__(mut self) -> String:
        """The next member name."""
        var key = self._src[]._member_key_at(self._index)
        self._index += 1
        return key^


struct ValueValuesIter[mut: Bool, //, origin: Origin[mut=mut]](
    Copyable, Iterator, Movable
):
    """Yields an object's member values. Built by `Value.values()`."""

    comptime Element = Value

    var _src: Pointer[Value, Self.origin]
    var _index: Int
    var _count: Int

    def __init__(out self, ref[Self.origin] src: Value):
        """Start at the first member of `src`."""
        self._src = Pointer(to=src)
        self._index = 0
        self._count = src.object_count()

    def __iter__(self) -> Self:
        """An iterator is its own iterable.

        Present so `for pair in obj.items():` reaches the loop
        protocol directly, the way a `for` loop over any other
        iterator does.
        """
        return self.copy()

    def __has_next__(self) -> Bool:
        """Whether any member remains."""
        return self._index < self._count

    def __next__(mut self) -> Value:
        """The next member value."""
        var value = self._src[]._member_value_at(self._index)
        self._index += 1
        return value^
