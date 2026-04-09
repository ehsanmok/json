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

from std.reflection import (
    struct_field_count,
    struct_field_names,
    struct_field_types,
    get_type_name,
    get_base_type_name,
    is_struct_type,
)
from std.builtin.rebind import trait_downcast, downcast
from std.collections import Optional, List

from .value import Value, Null
from .parser import loads
from .serialize import _escape_string
from .deserialize import get_string, get_int, get_bool, get_float

# ---------------------------------------------------------------------------
# Compile-time type name constants
# ---------------------------------------------------------------------------

comptime _INT_NAME = get_type_name[Int]()
comptime _INT64_NAME = get_type_name[Int64]()
comptime _BOOL_NAME = get_type_name[Bool]()
comptime _STRING_NAME = get_type_name[String]()
comptime _FLOAT64_NAME = get_type_name[Float64]()
comptime _FLOAT32_NAME = get_type_name[Float32]()
comptime _VALUE_NAME = get_type_name[Value]()

comptime _OPT_INT_NAME = get_type_name[Optional[Int]]()
comptime _OPT_STRING_NAME = get_type_name[Optional[String]]()
comptime _OPT_FLOAT64_NAME = get_type_name[Optional[Float64]]()
comptime _OPT_BOOL_NAME = get_type_name[Optional[Bool]]()

comptime _LIST_INT_NAME = get_type_name[List[Int]]()
comptime _LIST_STRING_NAME = get_type_name[List[String]]()
comptime _LIST_FLOAT64_NAME = get_type_name[List[Float64]]()
comptime _LIST_BOOL_NAME = get_type_name[List[Bool]]()

comptime _Base = ImplicitlyDestructible & Movable
comptime _JsonStruct = Defaultable & Movable


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
    """Serialize any struct to a JSON string via compile-time reflection.

    Parameters:
        T: The struct type (inferred).
        pretty: If True, format with 2-space indentation.

    Args:
        value: The struct instance to serialize.

    Returns:
        A JSON string representation.
    """
    var json = _ser[T](value)

    comptime if pretty:
        var parsed = loads(json)
        from .serialize import dumps as _dumps

        return _dumps(parsed, indent="  ")

    return json^


def serialize_value[T: AnyType](value: T) raises -> Value:
    """Serialize a struct to a json Value via compile-time reflection.

    Parameters:
        T: The struct type (inferred).

    Args:
        value: The struct instance.

    Returns:
        A json Value representing the JSON.
    """
    return loads(_ser[T](value))


# ===================================================================
# Public API -- Deserialization
# ===================================================================


def deserialize_json[T: _JsonStruct, target: StaticString = "cpu"](
    json_str: String,
) raises -> T:
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
    comptime
    if conforms_to(T, JsonDeserializable):
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


def try_deserialize_json[T: _JsonStruct, target: StaticString = "cpu"](
    json_str: String,
) -> Optional[T]:
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


def _ser[T: AnyType](value: T) raises -> String:
    """Dispatch serialization by compile-time type."""
    comptime tname = get_type_name[T]()

    comptime
    if tname == _STRING_NAME:
        return _escape_string(rebind[String](value))
    elif tname == _INT_NAME:
        return String(rebind[Int](value))
    elif tname == _INT64_NAME:
        return String(rebind[Int64](value))
    elif tname == _BOOL_NAME:
        return "true" if rebind[Bool](value) else "false"
    elif tname == _FLOAT64_NAME or "SIMD[DType.float64" in tname:
        return String(rebind[Float64](value))
    elif tname == _FLOAT32_NAME or "SIMD[DType.float32" in tname:
        return String(rebind[Float32](value))
    elif tname == _VALUE_NAME:
        return _ser_value(rebind[Value](value))
    elif tname == _OPT_INT_NAME:
        return _ser_opt_int(rebind[Optional[Int]](value))
    elif tname == _OPT_STRING_NAME:
        return _ser_opt_string(rebind[Optional[String]](value))
    elif tname == _OPT_FLOAT64_NAME:
        return _ser_opt_float64(rebind[Optional[Float64]](value))
    elif tname == _OPT_BOOL_NAME:
        return _ser_opt_bool(rebind[Optional[Bool]](value))
    elif tname == _LIST_INT_NAME:
        return _ser_list_int(rebind[List[Int]](value))
    elif tname == _LIST_STRING_NAME:
        return _ser_list_string(rebind[List[String]](value))
    elif tname == _LIST_FLOAT64_NAME:
        return _ser_list_float64(rebind[List[Float64]](value))
    elif tname == _LIST_BOOL_NAME:
        return _ser_list_bool(rebind[List[Bool]](value))
    elif is_struct_type[T]():
        comptime
        if conforms_to(T, JsonSerializable):
            ref custom = trait_downcast[JsonSerializable](value)
            var val = custom.to_json_value()
            return _ser_value(val)
        else:
            return _ser_struct[T](value)
    else:
        return "null"


def _ser_struct[T: AnyType](value: T) raises -> String:
    """Serialize a struct as ``{"field":value, ...}``."""
    comptime field_count = struct_field_count[T]()
    comptime field_names = struct_field_names[T]()
    comptime field_types = struct_field_types[T]()

    if field_count == 0:
        return "{}"

    var out = String("{")
    var first = True

    comptime
    for idx in range(field_count):
        if not first:
            out += ","
        first = False

        comptime field_name = field_names[idx]
        comptime field_type = field_types[idx]

        out += '"' + String(field_name) + '":'

        ref field = __struct_field_ref(idx, value)
        out += _ser[field_type](rebind[field_type](field))

    out += "}"
    return out^


# --- Value pass-through ---

def _ser_value(v: Value) -> String:
    if v.is_string():
        return _escape_string(v.string_value())
    elif v.is_null():
        return "null"
    elif v.is_bool():
        return "true" if v.bool_value() else "false"
    elif v.is_int():
        return String(v.int_value())
    elif v.is_float():
        return String(v.float_value())
    elif v.is_array() or v.is_object():
        return v.raw_json()
    return "null"


# --- Optional helpers ---

def _ser_opt_int(opt: Optional[Int]) -> String:
    if opt:
        return String(opt.value())
    return "null"


def _ser_opt_string(opt: Optional[String]) -> String:
    if opt:
        return _escape_string(opt.value())
    return "null"


def _ser_opt_float64(opt: Optional[Float64]) -> String:
    if opt:
        return String(opt.value())
    return "null"


def _ser_opt_bool(opt: Optional[Bool]) -> String:
    if opt:
        return "true" if opt.value() else "false"
    return "null"


# --- List helpers ---

def _ser_list_int(lst: List[Int]) -> String:
    var out = String("[")
    for i in range(len(lst)):
        if i > 0:
            out += ","
        out += String(lst[i])
    out += "]"
    return out^


def _ser_list_string(lst: List[String]) -> String:
    var out = String("[")
    for i in range(len(lst)):
        if i > 0:
            out += ","
        out += _escape_string(lst[i])
    out += "]"
    return out^


def _ser_list_float64(lst: List[Float64]) -> String:
    var out = String("[")
    for i in range(len(lst)):
        if i > 0:
            out += ","
        out += String(lst[i])
    out += "]"
    return out^


def _ser_list_bool(lst: List[Bool]) -> String:
    var out = String("[")
    for i in range(len(lst)):
        if i > 0:
            out += ","
        out += "true" if lst[i] else "false"
    out += "]"
    return out^


# ===================================================================
# Internal -- deserialization helpers
# ===================================================================


def _deser_fill[T: AnyType](mut result: T, json: Value) raises:
    """Fill every field of *result* from the JSON object *json*.

    Uses ``trait_downcast`` + ``UnsafePointer`` to write deserialized
    values into reflected struct fields. The struct must already be
    default-initialized; old field values are destroyed before writing.
    """
    comptime field_count = struct_field_count[T]()
    comptime field_names = struct_field_names[T]()
    comptime field_types = struct_field_types[T]()

    comptime
    for idx in range(field_count):
        comptime field_name = field_names[idx]
        comptime field_type = field_types[idx]
        comptime field_type_name = get_type_name[field_type]()
        var key = String(field_name)

        ref field = trait_downcast[_Base](__struct_field_ref(idx, result))
        var ptr = UnsafePointer(to=field)

        comptime
        if field_type_name == _STRING_NAME:
            ptr.destroy_pointee()
            ptr.bitcast[String]().init_pointee_move(get_string(json, key))
        elif field_type_name == _INT_NAME:
            ptr.destroy_pointee()
            ptr.bitcast[Int]().init_pointee_move(get_int(json, key))
        elif field_type_name == _INT64_NAME:
            ptr.destroy_pointee()
            var raw = json.get(key)
            var parsed = loads(raw)
            if not parsed.is_int():
                raise _field_type_error(key, "Int64", parsed)
            ptr.bitcast[Int64]().init_pointee_move(parsed.int_value())
        elif field_type_name == _BOOL_NAME:
            ptr.destroy_pointee()
            ptr.bitcast[Bool]().init_pointee_move(get_bool(json, key))
        elif field_type_name == _FLOAT64_NAME or "SIMD[DType.float64" in field_type_name:
            ptr.destroy_pointee()
            ptr.bitcast[Float64]().init_pointee_move(get_float(json, key))
        elif field_type_name == _FLOAT32_NAME or "SIMD[DType.float32" in field_type_name:
            ptr.destroy_pointee()
            ptr.bitcast[Float32]().init_pointee_move(
                Float32(get_float(json, key))
            )
        elif field_type_name == _VALUE_NAME:
            ptr.destroy_pointee()
            var raw = json.get(key)
            var v = loads(raw)
            ptr.bitcast[Value]().init_pointee_move(v^)
        # ----- Optional scalars -----
        elif field_type_name == _OPT_INT_NAME:
            ptr.destroy_pointee()
            ptr.bitcast[Optional[Int]]().init_pointee_move(
                _deser_opt_int(json, key)
            )
        elif field_type_name == _OPT_STRING_NAME:
            ptr.destroy_pointee()
            ptr.bitcast[Optional[String]]().init_pointee_move(
                _deser_opt_string(json, key)
            )
        elif field_type_name == _OPT_FLOAT64_NAME:
            ptr.destroy_pointee()
            ptr.bitcast[Optional[Float64]]().init_pointee_move(
                _deser_opt_float64(json, key)
            )
        elif field_type_name == _OPT_BOOL_NAME:
            ptr.destroy_pointee()
            ptr.bitcast[Optional[Bool]]().init_pointee_move(
                _deser_opt_bool(json, key)
            )
        # ----- List scalars -----
        elif field_type_name == _LIST_INT_NAME:
            ptr.destroy_pointee()
            ptr.bitcast[List[Int]]().init_pointee_move(
                _deser_list_int(json, key)
            )
        elif field_type_name == _LIST_STRING_NAME:
            ptr.destroy_pointee()
            ptr.bitcast[List[String]]().init_pointee_move(
                _deser_list_string(json, key)
            )
        elif field_type_name == _LIST_FLOAT64_NAME:
            ptr.destroy_pointee()
            ptr.bitcast[List[Float64]]().init_pointee_move(
                _deser_list_float64(json, key)
            )
        elif field_type_name == _LIST_BOOL_NAME:
            ptr.destroy_pointee()
            ptr.bitcast[List[Bool]]().init_pointee_move(
                _deser_list_bool(json, key)
            )
        # ----- Nested struct (fill existing default in-place) -----
        elif is_struct_type[field_type]():
            var raw = json.get(key)
            var sub_json = loads(raw)
            if not sub_json.is_object():
                raise _field_type_error(key, "object", sub_json)
            _deser_fill[field_type](ptr.bitcast[field_type]()[], sub_json)
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


def _deser_opt_float64(
    json: Value, key: String
) raises -> Optional[Float64]:
    if not _has_key(json, key) or _is_null_field(json, key):
        return None
    return get_float(json, key)


def _deser_opt_bool(json: Value, key: String) raises -> Optional[Bool]:
    if not _has_key(json, key) or _is_null_field(json, key):
        return None
    return get_bool(json, key)


# --- List deserialization ---


def _deser_list_int(json: Value, key: String) raises -> List[Int]:
    var raw = json.get(key)
    var arr = loads(raw)
    if not arr.is_array():
        raise _field_type_error(key, "array", arr)
    var items = arr.array_items()
    var result = List[Int]()
    for i in range(len(items)):
        if not items[i].is_int():
            raise Error(
                "Element " + String(i) + " of '" + key + "' expected int, got "
                + _type_label(items[i])
            )
        result.append(Int(items[i].int_value()))
    return result^


def _deser_list_string(json: Value, key: String) raises -> List[String]:
    var raw = json.get(key)
    var arr = loads(raw)
    if not arr.is_array():
        raise _field_type_error(key, "array", arr)
    var items = arr.array_items()
    var result = List[String]()
    for i in range(len(items)):
        if not items[i].is_string():
            raise Error(
                "Element " + String(i) + " of '" + key + "' expected string, got "
                + _type_label(items[i])
            )
        result.append(items[i].string_value())
    return result^


def _deser_list_float64(
    json: Value, key: String
) raises -> List[Float64]:
    var raw = json.get(key)
    var arr = loads(raw)
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
                "Element " + String(i) + " of '" + key + "' expected number, got "
                + _type_label(items[i])
            )
    return result^


def _deser_list_bool(json: Value, key: String) raises -> List[Bool]:
    var raw = json.get(key)
    var arr = loads(raw)
    if not arr.is_array():
        raise _field_type_error(key, "array", arr)
    var items = arr.array_items()
    var result = List[Bool]()
    for i in range(len(items)):
        if not items[i].is_bool():
            raise Error(
                "Element " + String(i) + " of '" + key + "' expected bool, got "
                + _type_label(items[i])
            )
        result.append(items[i].bool_value())
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
        var raw = json.get(key)
        return raw == "null"
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
        "Field '" + field + "' expected " + expected + ", got " + _type_label(got)
    )
