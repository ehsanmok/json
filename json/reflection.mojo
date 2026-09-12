"""Reflection-based JSON serialization and deserialization.

Zero-boilerplate serde for Mojo structs using compile-time reflection.
Structs are automatically mapped to/from JSON objects by reflecting over
field names and types at compile time.

Supported field types:
    Scalars: Int, Int64, Bool, Float64, Float32, String
    Containers: List[Int], List[String], List[Float64], List[Bool],
                Optional[Int], Optional[String], Optional[Float64], Optional[Bool]
    Nested structs (recursive reflection)
    Value (raw JSON pass-through)

Serialization requires no traits -- any struct works:

    @fieldwise_init
    struct Point:
        var x: Int
        var y: Int

    var json = serialize_json(Point(1, 2))  # {"x":1,"y":2}

Deserialization requires Defaultable and Movable:

    @fieldwise_init
    struct Point(Defaultable, Movable):
        var x: Int
        var y: Int

        def __init__(out self):
            self.x = 0
            self.y = 0

    var p = deserialize_json[Point]('{"x":1,"y":2}')
"""

from std.builtin.rebind import downcast
from std.collections import Optional, List, Dict

from .value import Value, Null
from .value.value import write_value
from .writer import JsonWriter
from .parser import loads
from .serialize import dumps
from .deserialize import get_string, get_int, get_bool, get_float

# ---------------------------------------------------------------------------
# Compile-time type name constants
# ---------------------------------------------------------------------------

comptime _INT_NAME = reflect[Int].name()
comptime _INT64_NAME = reflect[Int64].name()
comptime _INT32_NAME = reflect[Int32].name()
comptime _INT16_NAME = reflect[Int16].name()
comptime _INT8_NAME = reflect[Int8].name()
comptime _UINT64_NAME = reflect[UInt64].name()
comptime _UINT32_NAME = reflect[UInt32].name()
comptime _UINT16_NAME = reflect[UInt16].name()
comptime _UINT8_NAME = reflect[UInt8].name()
comptime _BOOL_NAME = reflect[Bool].name()
comptime _STRING_NAME = reflect[String].name()
comptime _VALUE_NAME = reflect[Value].name()

comptime _OPT_INT_NAME = reflect[Optional[Int]].name()
comptime _OPT_STRING_NAME = reflect[Optional[String]].name()
comptime _OPT_FLOAT64_NAME = reflect[Optional[Float64]].name()
comptime _OPT_BOOL_NAME = reflect[Optional[Bool]].name()

comptime _LIST_INT_NAME = reflect[List[Int]].name()
comptime _LIST_STRING_NAME = reflect[List[String]].name()
comptime _LIST_FLOAT64_NAME = reflect[List[Float64]].name()
comptime _LIST_BOOL_NAME = reflect[List[Bool]].name()

# Composite reflected types: Dict[String, T], nested Lists, Optional<->List combos.
comptime _DICT_STRING_INT_NAME = reflect[Dict[String, Int]].name()
comptime _DICT_STRING_STRING_NAME = reflect[Dict[String, String]].name()
comptime _DICT_STRING_FLOAT64_NAME = reflect[Dict[String, Float64]].name()
comptime _DICT_STRING_BOOL_NAME = reflect[Dict[String, Bool]].name()

comptime _LIST_OPT_INT_NAME = reflect[List[Optional[Int]]].name()
comptime _LIST_OPT_STRING_NAME = reflect[List[Optional[String]]].name()

comptime _OPT_LIST_INT_NAME = reflect[Optional[List[Int]]].name()
comptime _OPT_LIST_STRING_NAME = reflect[Optional[List[String]]].name()

comptime _LIST_LIST_INT_NAME = reflect[List[List[Int]]].name()
comptime _LIST_LIST_STRING_NAME = reflect[List[List[String]]].name()

comptime _Base = Deinitable & Movable
comptime _JsonStruct = Defaultable & Movable & Deinitable


# ===================================================================
# Generic container emission
# ===================================================================
#
# Dispatch is by type through a trait, not by the spelling of a type's
# name. The ladder this replaces matched `reflect[T].name()` against a
# constant per supported type, which meant a new constant and a new arm
# for every combination, and one of its comparisons was a substring
# test that made `List[Float64]` fail to compile.
#
# The obstacle a name ladder was working around is real: inside a
# function parametric on `T`, a reflected field type stays symbolic --
# bound only by `AnyType` -- so `E` cannot be deduced from `List[E]`,
# no `[E](List[E])` overload matches, and even `len()` does not
# resolve.
#
# Retroactive conformance solves it. Inside an extension body, the
# container's own element parameter is concrete for each
# instantiation, so an ordinary generic call deduces the element type
# by argument deduction, and `_ser_into` reaches it through
# `conforms_to` plus `downcast` without ever naming `E`. That is what
# makes `List[<struct>]`, `Optional[List[Int]]` and
# `Dict[String, <struct>]` work through one path.
#
# Ordering is the one subtlety: `List`, `Optional` and `Dict` are
# themselves structs, so the conformance check has to run before
# `reflect[T].is_struct()`, or a list would be serialized as an object
# holding its data pointer, length and capacity.


trait _JsonEmit:
    """Emit self as JSON into a writer. Internal."""

    def emit_json(self, mut w: JsonWriter) raises:
        ...


__extension String(_JsonEmit):
    def emit_json(self, mut w: JsonWriter) raises:
        w.write_string_span(self.as_bytes())


__extension Bool(_JsonEmit):
    def emit_json(self, mut w: JsonWriter) raises:
        w.write_bool(self)


__extension SIMD(_JsonEmit):
    def emit_json(self, mut w: JsonWriter) raises:
        """Every numeric width through one arm.

        A `Float64` field reflects under its canonical SIMD spelling,
        as do `Int32`, `UInt8` and the rest, so matching them by name
        meant a constant per width and a ladder to walk. The element
        type is a parameter here, so one branch on what it is covers
        all of them, and a wider vector -- which JSON has no scalar
        spelling for -- becomes an array of its lanes.
        """
        comptime if Self.length == 1:
            comptime if Self.dtype.is_floating_point():
                w.write_float(Float64(self[0]))
            elif Self.dtype.is_signed():
                w.write_int(Int64(self[0]))
            else:
                w.write_uint(UInt64(self[0]))
        else:
            w.write_byte(UInt8(0x5B))
            comptime for lane in range(Self.length):
                comptime if lane > 0:
                    w.write_byte(UInt8(0x2C))
                comptime if Self.dtype.is_floating_point():
                    w.write_float(Float64(self[lane]))
                elif Self.dtype.is_signed():
                    w.write_int(Int64(self[lane]))
                else:
                    w.write_uint(UInt64(self[lane]))
            w.write_byte(UInt8(0x5D))


__extension List(_JsonEmit):
    def emit_json(self, mut w: JsonWriter) raises:
        w.write_byte(UInt8(0x5B))
        for i in range(len(self)):
            if i > 0:
                w.write_byte(UInt8(0x2C))
            _ser_into(w, self[i])
        w.write_byte(UInt8(0x5D))


__extension Optional(_JsonEmit):
    def emit_json(self, mut w: JsonWriter) raises:
        if self:
            _ser_into(w, self.value())
        else:
            w.write_null()


__extension Dict(_JsonEmit):
    def emit_json(self, mut w: JsonWriter) raises:
        w.write_byte(UInt8(0x7B))
        var first = True
        for entry in self.items():
            if not first:
                w.write_byte(UInt8(0x2C))
            first = False
            comptime if Self.K == String:
                w.write_string_span(rebind[String](entry.key).as_bytes())
            else:
                comptime assert False, (
                    "a JSON object's keys are strings; Dict[K, V] needs K ="
                    " String"
                )
            w.write_byte(UInt8(0x3A))
            _ser_into(w, entry.value)
        w.write_byte(UInt8(0x7D))


__extension Value(_JsonEmit):
    def emit_json(self, mut w: JsonWriter) raises:
        write_value(w, self)


# ===================================================================
# Custom serde traits
# ===================================================================


trait JsonSerializable:
    """Override reflection serialization for a struct.

    Implement this to control exactly how a struct is serialized to a
    json Value. The reflection serializer will call ``to_json_value``
    instead of walking fields.

    Example::

        @fieldwise_init
        struct Color(JsonSerializable, Defaultable, Movable):
            var r: Int
            var g: Int
            var b: Int

            def to_json_value(self) raises -> Value:
                return loads(
                    '"rgb(' + String(self.r) + ","
                    + String(self.g) + "," + String(self.b) + ')"'
                )
    """

    def to_json_value(self) raises -> Value:
        ...


trait JsonDeserializable:
    """Override reflection deserialization for a struct.

    Implement this to control exactly how a struct is deserialized from
    a json Value. The reflection deserializer will call
    ``from_json_value`` instead of walking fields.

    Example::

        @fieldwise_init
        struct Color(JsonDeserializable, Defaultable, Movable):
            var r: Int
            var g: Int
            var b: Int

            @staticmethod
            def from_json_value(json: Value) raises -> Self:
                var s = json.string_value()
                # parse "rgb(r,g,b)" ...
                return Self(r=..., g=..., b=...)
    """

    @staticmethod
    def from_json_value(json: Value) raises -> Self:
        ...


# ===================================================================
# Public API -- Serialization
# ===================================================================


def serialize_json[T: AnyType, pretty: Bool = False](value: T) raises -> String:
    """Serialize any value to JSON through compile-time reflection.

    Indentation is a second pass over the compact output, on purpose:
    making it a mode of the first pass turns every separator in the
    comptime-unrolled emitter into a runtime branch and costs the
    compact path, which is the one that runs in anger.

    Parameters:
        T: The type, inferred from the argument.
        pretty: Indent with two spaces per level.

    Args:
        value: The value to serialize.

    Returns:
        The JSON text.

    Raises:
        If a custom `to_json_value` raises.
    """
    var w = JsonWriter(capacity=256)
    _ser_into[T](w, value)
    comptime if pretty:
        return dumps(loads(w^.finish_string()), indent="  ")
    return w^.finish_string()


def serialize_json_into[T: AnyType](mut w: JsonWriter, value: T) raises:
    """Serialize into a writer the caller owns.

    For a caller emitting many values: the buffer is the expensive
    part, and `reset` makes it reusable. Also the way to splice a
    typed value into a larger document without it becoming its own
    `String` first.

    Parameters:
        T: The type, inferred from the argument.

    Args:
        w: The writer to emit into.
        value: The value to serialize.

    Raises:
        If a custom `to_json_value` raises.
    """
    _ser_into[T](w, value)


def serialize_value[T: AnyType](value: T) raises -> Value:
    """Serialize a struct to a json Value via compile-time reflection.

    Parameters:
        T: The struct type (inferred).

    Args:
        value: The struct instance.

    Returns:
        A json Value representing the JSON.
    """
    return loads(serialize_json[T](value))


# ===================================================================
# Public API -- Deserialization
# ===================================================================


def deserialize_json[
    T: _JsonStruct, target: StaticString = "cpu"
](json_str: String,) raises -> T:
    """Deserialize a JSON string into a struct via compile-time reflection.

    Uses ``out``-parameter initialization so the struct does **not** need
    ``Defaultable``; only ``Movable`` is required.

    Parameters:
        T: The target struct type.
        target: Parsing backend (``"cpu"`` or ``"gpu"``).

    Args:
        json_str: The JSON string.

    Returns:
        A populated struct of type T.

    Raises:
        Error on parse failure, missing required fields, or type mismatches.
    """
    var json = loads[target](json_str)
    return deserialize_value[T](json)


def deserialize_value[T: _JsonStruct](json: Value) raises -> T:
    """Deserialize a json Value into a struct via compile-time reflection.

    If ``T`` conforms to ``JsonDeserializable``, the custom
    ``from_json_value`` is called instead of walking fields.

    Parameters:
        T: The target struct type (Defaultable & Movable).

    Args:
        json: A json Value (must be a JSON object).

    Returns:
        A populated struct.
    """
    comptime if conforms_to(T, JsonDeserializable):
        return downcast[T, JsonDeserializable].from_json_value(json)
    else:
        if not json.is_object():
            raise Error(
                "Expected JSON object for struct deserialization, got "
                + _type_label(json)
            )
        var result = T()
        _deser_fill[T](result, json)
        return result^


def try_deserialize_json[
    T: _JsonStruct, target: StaticString = "cpu"
](json_str: String,) -> Optional[T]:
    """Non-raising variant of ``deserialize_json``.

    Parameters:
        T: The target struct type.
        target: Parsing backend.

    Args:
        json_str: The JSON string.

    Returns:
        ``Optional`` containing the struct, or ``None`` on any error.
    """
    try:
        return deserialize_json[T, target](json_str)
    except:
        return None


# ===================================================================
# Internal -- serialization helpers
# ===================================================================


@always_inline
def _ser_into[T: AnyType](mut w: JsonWriter, value: T) raises:
    """Emit `value` as JSON into `w`.

    Dispatch is by type, not by the spelling of a type's name. The
    ladder this replaces matched `reflect[T].name()` against a
    constant per supported type, which meant every new combination
    needed a new constant and a new arm, `Optional` and `Dict` fields
    fell through to a path that built a `String` and copied it back
    in, and one of the comparisons was a substring test that made
    `List[Float64]` unusable.

    The scalar arms up front are not redundant with the `_JsonEmit`
    conformance below, which also covers them. Reaching a scalar
    through the trait costs an indirect call, and a record is mostly
    scalars -- routing them through it made the benchmark's document
    shape 45% slower. These arms compare types, not type names, so
    the substring test that used to make `List[Float64]` fail cannot
    come back.

    Order matters after that. A custom `to_json_value` wins over the
    default, so a type can override its own representation. And
    `_JsonEmit` has to come before `reflect[T].is_struct()`, because
    `List`, `Optional` and `Dict` are themselves structs and would
    otherwise be serialized field by field -- which is how a list once
    emitted its data pointer, length and capacity as a JSON object.
    """
    comptime if T == String:
        w.write_string(rebind[String](value))
    elif T == Int:
        w.write_int(Int64(rebind[Int](value)))
    elif T == Bool:
        w.write_bool(rebind[Bool](value))
    elif T == Float64:
        w.write_float(rebind[Float64](value))
    elif T == Int64:
        w.write_int(rebind[Int64](value))
    elif T == Int32:
        w.write_int(Int64(rebind[Int32](value)))
    elif T == UInt64:
        w.write_uint(rebind[UInt64](value))
    elif T == UInt:
        w.write_uint(UInt64(rebind[UInt](value)))
    elif T == Float32:
        w.write_float(Float64(rebind[Float32](value)))
    elif T == Int16:
        w.write_int(Int64(rebind[Int16](value)))
    elif T == Int8:
        w.write_int(Int64(rebind[Int8](value)))
    elif T == UInt32:
        w.write_uint(UInt64(rebind[UInt32](value)))
    elif T == UInt16:
        w.write_uint(UInt64(rebind[UInt16](value)))
    elif T == UInt8:
        w.write_uint(UInt64(rebind[UInt8](value)))
    elif conforms_to(T, JsonSerializable):
        ref custom = rebind[downcast[T, JsonSerializable]](value)
        write_value(w, custom.to_json_value())
    elif conforms_to(T, _JsonEmit):
        rebind[downcast[T, _JsonEmit]](value).emit_json(w)
    elif reflect[T].is_struct():
        _ser_struct_into[T](w, value)
    else:
        comptime assert False, (
            "serialize_json: unsupported field type " + reflect[T].name()
        )


def _ser_struct_into[T: AnyType](mut w: JsonWriter, value: T) raises:
    """Emit a struct as an object, unrolled at compile time.

    The field count and names are comptime, so the separator decision
    is too -- there is no runtime comma branching. The key is emitted
    from the comptime name's bytes directly: a struct field name is a
    Mojo identifier, so it never needs escaping, and building a
    `String` for it would allocate once per field per record.

    Separators are emitted under a comptime branch, so an unrolled
    struct costs no runtime comma decision at all. Routing them
    through the writer's indent-aware container helpers instead turned
    each one into a runtime branch and made this shape 45% slower --
    which is why indentation is a separate pass rather than a mode of
    this one.
    """
    comptime field_count = reflect[T].field_count()
    comptime field_names = reflect[T].field_names()
    comptime field_types = reflect[T].field_types()

    w.write_byte(UInt8(0x7B))
    comptime for idx in range(field_count):
        comptime if idx > 0:
            w.write_byte(UInt8(0x2C))
        comptime field_name = field_names[idx]
        comptime field_type = field_types[idx]
        w.write_byte(UInt8(0x22))
        w.write_bytes(field_name.as_bytes())
        w.write_byte(UInt8(0x22))
        w.write_byte(UInt8(0x3A))
        ref field = reflect[T].field_ref[idx](value)
        _ser_into[field_type](w, rebind[field_type](field))
    w.write_byte(UInt8(0x7D))


# ===================================================================
# Internal -- deserialization helpers
# ===================================================================


def _get_sized_int(json: Value, key: String, type_name: String) raises -> Int64:
    """Read an integer field as `Int64`, for the sized-integer arms.

    Factored out of the `Int64` arm so every width shares one code path
    and one error message. Narrower widths convert at the call site; the
    value is not range-checked, matching the existing `Int64` behaviour.

    Reads the child `Value` directly. This used to serialize the child
    subtree to a raw-JSON `String` and run a full parse over it to
    recover one integer.
    """
    var parsed = json[key]
    if not parsed.is_int():
        raise _field_type_error(key, type_name, parsed)
    return parsed.int_value()


def _deser_fill[T: AnyType](mut result: T, json: Value) raises:
    """Fill every field of *result* from the JSON object *json*.

    Uses ``downcast`` + ``Pointer`` to write deserialized
    values into reflected struct fields. The struct must already be
    default-initialized; old field values are destroyed before writing.
    """
    comptime field_count = reflect[T].field_count()
    comptime field_names = reflect[T].field_names()
    comptime field_types = reflect[T].field_types()

    comptime for idx in range(field_count):
        comptime field_name = field_names[idx]
        comptime field_type = field_types[idx]
        comptime field_type_name = reflect[field_type].name()
        var key = String(field_name)

        ref field = rebind[downcast[field_type, _Base]](
            reflect[T].field_ref[idx](result)
        )
        var ptr = Pointer(to=field)

        comptime if field_type_name == _STRING_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[String]().unsafe_write(get_string(json, key))
        elif field_type_name == _INT_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Int]().unsafe_write(get_int(json, key))
        elif field_type_name == _INT64_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Int64]().unsafe_write(
                _get_sized_int(json, key, "Int64")
            )
        elif field_type_name == _INT32_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Int32]().unsafe_write(
                Int32(_get_sized_int(json, key, "Int32"))
            )
        elif field_type_name == _INT16_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Int16]().unsafe_write(
                Int16(_get_sized_int(json, key, "Int16"))
            )
        elif field_type_name == _INT8_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Int8]().unsafe_write(
                Int8(_get_sized_int(json, key, "Int8"))
            )
        elif field_type_name == _UINT64_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[UInt64]().unsafe_write(
                UInt64(_get_sized_int(json, key, "UInt64"))
            )
        elif field_type_name == _UINT32_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[UInt32]().unsafe_write(
                UInt32(_get_sized_int(json, key, "UInt32"))
            )
        elif field_type_name == _UINT16_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[UInt16]().unsafe_write(
                UInt16(_get_sized_int(json, key, "UInt16"))
            )
        elif field_type_name == _UINT8_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[UInt8]().unsafe_write(
                UInt8(_get_sized_int(json, key, "UInt8"))
            )
        elif field_type_name == _BOOL_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Bool]().unsafe_write(get_bool(json, key))
        elif field_type == Float64:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Float64]().unsafe_write(get_float(json, key))
        elif field_type == Float32:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Float32]().unsafe_write(
                Float32(get_float(json, key))
            )
        elif field_type_name == _VALUE_NAME:
            ptr.unsafe_deinit_pointee()
            var v = json[key]
            ptr.unsafe_bitcast[Value]().unsafe_write(v^)
        # ----- Optional scalars -----
        elif field_type_name == _OPT_INT_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Optional[Int]]().unsafe_write(
                _deser_opt_int(json, key)
            )
        elif field_type_name == _OPT_STRING_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Optional[String]]().unsafe_write(
                _deser_opt_string(json, key)
            )
        elif field_type_name == _OPT_FLOAT64_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Optional[Float64]]().unsafe_write(
                _deser_opt_float64(json, key)
            )
        elif field_type_name == _OPT_BOOL_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Optional[Bool]]().unsafe_write(
                _deser_opt_bool(json, key)
            )
        # ----- List scalars -----
        elif field_type_name == _LIST_INT_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[List[Int]]().unsafe_write(
                _deser_list_int(json, key)
            )
        elif field_type_name == _LIST_STRING_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[List[String]]().unsafe_write(
                _deser_list_string(json, key)
            )
        elif field_type_name == _LIST_FLOAT64_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[List[Float64]]().unsafe_write(
                _deser_list_float64(json, key)
            )
        elif field_type_name == _LIST_BOOL_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[List[Bool]]().unsafe_write(
                _deser_list_bool(json, key)
            )
        # ----- Combinator types: Dict, nested List, Optional<->List combos. -----
        elif field_type_name == _DICT_STRING_INT_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Dict[String, Int]]().unsafe_write(
                _deser_dict_string_int(json, key)
            )
        elif field_type_name == _DICT_STRING_STRING_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Dict[String, String]]().unsafe_write(
                _deser_dict_string_string(json, key)
            )
        elif field_type_name == _DICT_STRING_FLOAT64_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Dict[String, Float64]]().unsafe_write(
                _deser_dict_string_float64(json, key)
            )
        elif field_type_name == _DICT_STRING_BOOL_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Dict[String, Bool]]().unsafe_write(
                _deser_dict_string_bool(json, key)
            )
        elif field_type_name == _LIST_OPT_INT_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[List[Optional[Int]]]().unsafe_write(
                _deser_list_opt_int(json, key)
            )
        elif field_type_name == _LIST_OPT_STRING_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[List[Optional[String]]]().unsafe_write(
                _deser_list_opt_string(json, key)
            )
        elif field_type_name == _OPT_LIST_INT_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Optional[List[Int]]]().unsafe_write(
                _deser_opt_list_int(json, key)
            )
        elif field_type_name == _OPT_LIST_STRING_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[Optional[List[String]]]().unsafe_write(
                _deser_opt_list_string(json, key)
            )
        elif field_type_name == _LIST_LIST_INT_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[List[List[Int]]]().unsafe_write(
                _deser_list_list_int(json, key)
            )
        elif field_type_name == _LIST_LIST_STRING_NAME:
            ptr.unsafe_deinit_pointee()
            ptr.unsafe_bitcast[List[List[String]]]().unsafe_write(
                _deser_list_list_string(json, key)
            )
        elif field_type_name.startswith("List["):
            # A `List` is itself a struct, so without this arm a
            # `List[MyStruct]` field reached the nested-struct arm below
            # and failed with the misleading "expected object, got
            # array". The serialize direction now handles any element
            # type via retroactive conformance (see `_JsonEmit`); the
            # read direction needs more than emission -- it has to
            # construct and fill the list -- so it still declines
            # explicitly rather than guessing.
            raise Error(
                "deserialize_json: unsupported list element type for '"
                + key
                + "' ("
                + String(field_type_name)
                + "). Serialization handles any list element type; the"
                + " deserialize side does not yet, because filling a"
                + " List[E] in place needs E to be default-constructible"
                + " and writable through a reflected field pointer."
                + " Reflection covers List[Int|Int64|String|Float64|Bool],"
                + " List[Optional[Int|String]] and List[List[Int|String]];"
                + " for other element types implement JsonDeserializable"
                + " on the containing struct."
            )
        # ----- Nested struct (fill existing default in-place) -----
        elif reflect[field_type].is_struct():
            var sub_json = json[key]
            if not sub_json.is_object():
                raise _field_type_error(key, "object", sub_json)
            _deser_fill[field_type](
                ptr.unsafe_bitcast[field_type]()[], sub_json
            )
        else:
            raise Error(
                "Unsupported field type for '"
                + key
                + "': "
                + String(field_type_name)
            )


# --- Optional deserialization ---


def _deser_opt_int(json: Value, key: String) raises -> Optional[Int]:
    if not _has_key(json, key) or _is_null_field(json, key):
        return None
    return get_int(json, key)


def _deser_opt_string(json: Value, key: String) raises -> Optional[String]:
    if not _has_key(json, key) or _is_null_field(json, key):
        return None
    return get_string(json, key)


def _deser_opt_float64(json: Value, key: String) raises -> Optional[Float64]:
    if not _has_key(json, key) or _is_null_field(json, key):
        return None
    return get_float(json, key)


def _deser_opt_bool(json: Value, key: String) raises -> Optional[Bool]:
    if not _has_key(json, key) or _is_null_field(json, key):
        return None
    return get_bool(json, key)


# --- List deserialization ---


def _deser_list_int(json: Value, key: String) raises -> List[Int]:
    var arr = json[key]
    if not arr.is_array():
        raise _field_type_error(key, "array", arr)
    var items = arr.array_items()
    var result = List[Int]()
    for i in range(len(items)):
        if not items[i].is_int():
            raise Error(
                "Element "
                + String(i)
                + " of '"
                + key
                + "' expected int, got "
                + _type_label(items[i])
            )
        result.append(Int(items[i].int_value()))
    return result^


def _deser_list_string(json: Value, key: String) raises -> List[String]:
    var arr = json[key]
    if not arr.is_array():
        raise _field_type_error(key, "array", arr)
    var items = arr.array_items()
    var result = List[String]()
    for i in range(len(items)):
        if not items[i].is_string():
            raise Error(
                "Element "
                + String(i)
                + " of '"
                + key
                + "' expected string, got "
                + _type_label(items[i])
            )
        result.append(items[i].string_value())
    return result^


def _deser_list_float64(json: Value, key: String) raises -> List[Float64]:
    var arr = json[key]
    if not arr.is_array():
        raise _field_type_error(key, "array", arr)
    var items = arr.array_items()
    var result = List[Float64]()
    for i in range(len(items)):
        if items[i].is_float():
            result.append(items[i].float_value())
        elif items[i].is_int():
            result.append(Float64(items[i].int_value()))
        else:
            raise Error(
                "Element "
                + String(i)
                + " of '"
                + key
                + "' expected number, got "
                + _type_label(items[i])
            )
    return result^


def _deser_list_bool(json: Value, key: String) raises -> List[Bool]:
    var arr = json[key]
    if not arr.is_array():
        raise _field_type_error(key, "array", arr)
    var items = arr.array_items()
    var result = List[Bool]()
    for i in range(len(items)):
        if not items[i].is_bool():
            raise Error(
                "Element "
                + String(i)
                + " of '"
                + key
                + "' expected bool, got "
                + _type_label(items[i])
            )
        result.append(items[i].bool_value())
    return result^


# --- Dict[String, T] deserialization ---


def _deser_dict_string_int(
    json: Value, key: String
) raises -> Dict[String, Int]:
    var obj = json[key]
    if not obj.is_object():
        raise _field_type_error(key, "object", obj)
    var result = Dict[String, Int]()
    var keys = obj.object_keys()
    for i in range(len(keys)):
        var k = keys[i]
        var v = obj[k]
        if not v.is_int():
            raise Error(
                "Value at '"
                + key
                + "."
                + k
                + "' expected int, got "
                + _type_label(v)
            )
        result[k] = Int(v.int_value())
    return result^


def _deser_dict_string_string(
    json: Value, key: String
) raises -> Dict[String, String]:
    var obj = json[key]
    if not obj.is_object():
        raise _field_type_error(key, "object", obj)
    var result = Dict[String, String]()
    var keys = obj.object_keys()
    for i in range(len(keys)):
        var k = keys[i]
        var v = obj[k]
        if not v.is_string():
            raise Error(
                "Value at '"
                + key
                + "."
                + k
                + "' expected string, got "
                + _type_label(v)
            )
        result[k] = v.string_value()
    return result^


def _deser_dict_string_float64(
    json: Value, key: String
) raises -> Dict[String, Float64]:
    var obj = json[key]
    if not obj.is_object():
        raise _field_type_error(key, "object", obj)
    var result = Dict[String, Float64]()
    var keys = obj.object_keys()
    for i in range(len(keys)):
        var k = keys[i]
        var v = obj[k]
        if v.is_float():
            result[k] = v.float_value()
        elif v.is_int():
            result[k] = Float64(v.int_value())
        else:
            raise Error(
                "Value at '"
                + key
                + "."
                + k
                + "' expected number, got "
                + _type_label(v)
            )
    return result^


def _deser_dict_string_bool(
    json: Value, key: String
) raises -> Dict[String, Bool]:
    var obj = json[key]
    if not obj.is_object():
        raise _field_type_error(key, "object", obj)
    var result = Dict[String, Bool]()
    var keys = obj.object_keys()
    for i in range(len(keys)):
        var k = keys[i]
        var v = obj[k]
        if not v.is_bool():
            raise Error(
                "Value at '"
                + key
                + "."
                + k
                + "' expected bool, got "
                + _type_label(v)
            )
        result[k] = v.bool_value()
    return result^


# --- List[Optional[T]] deserialization ---


def _deser_list_opt_int(json: Value, key: String) raises -> List[Optional[Int]]:
    var arr = json[key]
    if not arr.is_array():
        raise _field_type_error(key, "array", arr)
    var items = arr.array_items()
    var result = List[Optional[Int]]()
    for i in range(len(items)):
        if items[i].is_null():
            result.append(None)
        elif items[i].is_int():
            result.append(Int(items[i].int_value()))
        else:
            raise Error(
                "Element "
                + String(i)
                + " of '"
                + key
                + "' expected int or null, got "
                + _type_label(items[i])
            )
    return result^


def _deser_list_opt_string(
    json: Value, key: String
) raises -> List[Optional[String]]:
    var arr = json[key]
    if not arr.is_array():
        raise _field_type_error(key, "array", arr)
    var items = arr.array_items()
    var result = List[Optional[String]]()
    for i in range(len(items)):
        if items[i].is_null():
            result.append(None)
        elif items[i].is_string():
            result.append(items[i].string_value())
        else:
            raise Error(
                "Element "
                + String(i)
                + " of '"
                + key
                + "' expected string or null, got "
                + _type_label(items[i])
            )
    return result^


# --- Optional[List[T]] deserialization ---


def _deser_opt_list_int(json: Value, key: String) raises -> Optional[List[Int]]:
    if not _has_key(json, key) or _is_null_field(json, key):
        return None
    return _deser_list_int(json, key)


def _deser_opt_list_string(
    json: Value, key: String
) raises -> Optional[List[String]]:
    if not _has_key(json, key) or _is_null_field(json, key):
        return None
    return _deser_list_string(json, key)


# --- List[List[T]] deserialization ---


def _deser_list_list_int(json: Value, key: String) raises -> List[List[Int]]:
    var arr = json[key]
    if not arr.is_array():
        raise _field_type_error(key, "array", arr)
    var outer = arr.array_items()
    var result = List[List[Int]]()
    for i in range(len(outer)):
        if not outer[i].is_array():
            raise Error(
                "Element "
                + String(i)
                + " of '"
                + key
                + "' expected array, got "
                + _type_label(outer[i])
            )
        var inner = outer[i].array_items()
        var row = List[Int]()
        for j in range(len(inner)):
            if not inner[j].is_int():
                raise Error(
                    "Element ["
                    + String(i)
                    + "]["
                    + String(j)
                    + "] of '"
                    + key
                    + "' expected int, got "
                    + _type_label(inner[j])
                )
            row.append(Int(inner[j].int_value()))
        result.append(row^)
    return result^


def _deser_list_list_string(
    json: Value, key: String
) raises -> List[List[String]]:
    var arr = json[key]
    if not arr.is_array():
        raise _field_type_error(key, "array", arr)
    var outer = arr.array_items()
    var result = List[List[String]]()
    for i in range(len(outer)):
        if not outer[i].is_array():
            raise Error(
                "Element "
                + String(i)
                + " of '"
                + key
                + "' expected array, got "
                + _type_label(outer[i])
            )
        var inner = outer[i].array_items()
        var row = List[String]()
        for j in range(len(inner)):
            if not inner[j].is_string():
                raise Error(
                    "Element ["
                    + String(i)
                    + "]["
                    + String(j)
                    + "] of '"
                    + key
                    + "' expected string, got "
                    + _type_label(inner[j])
                )
            row.append(inner[j].string_value())
        result.append(row^)
    return result^


# ===================================================================
# Utilities
# ===================================================================


def _has_key(json: Value, key: String) -> Bool:
    """Check whether a JSON object contains *key*."""
    if not json.is_object():
        return False
    var keys = json.object_keys()
    for i in range(len(keys)):
        if keys[i] == key:
            return True
    return False


def _is_null_field(json: Value, key: String) -> Bool:
    """Return True if the field is missing or its raw value is ``null``."""
    try:
        return json[key].is_null()
    except:
        return True


def _type_label(v: Value) -> String:
    """Human-readable label for the JSON type of *v*."""
    if v.is_null():
        return "null"
    elif v.is_bool():
        return "bool"
    elif v.is_int():
        return "int"
    elif v.is_float():
        return "float"
    elif v.is_string():
        return "string"
    elif v.is_array():
        return "array"
    elif v.is_object():
        return "object"
    return "unknown"


def _field_type_error(field: String, expected: String, got: Value) -> Error:
    """Build a descriptive error for a type mismatch on a field."""
    return Error(
        "Field '"
        + field
        + "' expected "
        + expected
        + ", got "
        + _type_label(got)
    )
