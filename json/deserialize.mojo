# json - JSON field extraction helpers
#
# These read a field straight off the parsed `Value`. They used to go
# `value.get(key)` -> a raw-JSON `String` of the child subtree -> a full
# `loads` of that string -> read one scalar. That is a serialize plus a
# complete SIMD parse per field, and for a nested struct it repeated at
# every level of depth.

from .value import Value
from .parser import loads


def get_string(value: Value, key: String) raises -> String:
    """Extract a string field from a JSON object Value.

    Args:
        value: The JSON object Value.
        key: The field name.

    Returns:
        The string value.

    Example:
        var json = loads('{"name": "Alice"}')
        var name = get_string(json, "name")  # Returns "Alice".
    """
    var field = value[key]
    if not field.is_string():
        raise Error("Field '" + key + "' is not a string")
    return field.string_value()


def get_int(value: Value, key: String) raises -> Int:
    """Extract an int field from a JSON object Value.

    Args:
        value: The JSON object Value.
        key: The field name.

    Returns:
        The int value.

    Example:
        var json = loads('{"age": 30}')
        var age = get_int(json, "age")  # Returns 30.
    """
    var field = value[key]
    if not field.is_int():
        raise Error("Field '" + key + "' is not an int")
    return Int(field.int_value())


def get_bool(value: Value, key: String) raises -> Bool:
    """Extract a bool field from a JSON object Value.

    Args:
        value: The JSON object Value.
        key: The field name.

    Returns:
        The bool value.

    Example:
        var json = loads('{"active": true}')
        var active = get_bool(json, "active")  # Returns True.
    """
    var field = value[key]
    if not field.is_bool():
        raise Error("Field '" + key + "' is not a bool")
    return field.bool_value()


def get_float(value: Value, key: String) raises -> Float64:
    """Extract a float field from a JSON object Value.

    Args:
        value: The JSON object Value.
        key: The field name.

    Returns:
        The float value.

    Example:
        var json = loads('{"price": 19.99}')
        var price = get_float(json, "price")  # Returns 19.99.
    """
    var field = value[key]
    if field.is_float():
        return field.float_value()
    elif field.is_int():
        return Float64(field.int_value())
    else:
        raise Error("Field '" + key + "' is not a number")


trait Deserializable:
    """Trait for types that can be deserialized from JSON.

    Implement this trait to enable automatic deserialization with deserialize().

    Example:
        struct Person(Deserializable):
            var name: String
            var age: Int

            @staticmethod
            def from_json(json: Value) raises -> Self:
                return Self(
                    name=get_string(json, "name"),
                    age=get_int(json, "age")
                )

        var person = deserialize[Person]('{"name":"Alice","age":30}')
    """

    @staticmethod
    def from_json(json: Value) raises -> Self:
        """Deserialize from a JSON Value."""
        ...


def deserialize[
    T: Deserializable, target: StaticString = "cpu"
](json_str: String) raises -> T:
    """Deserialize a JSON string into a typed object.

    The type must implement the Deserializable trait with a from_json() static method.

    Parameters:
        T: Type that implements Deserializable.
        target: "cpu" (default) or "gpu" for parsing backend.

    Args:
        json_str: JSON string to deserialize.

    Returns:
        Deserialized object of type T.

    Example:
        var person = deserialize[Person]('{"name":"Alice","age":30}')
        print(person.name)  # Prints "Alice".
    """
    var json = loads[target](json_str)
    return T.from_json(json)
