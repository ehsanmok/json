# mojson - JSON field extraction helpers

from .value import Value
from .parser import loads


fn get_string(value: Value, key: String) raises -> String:
    """Extract a string field from a JSON object Value.
    
    Args:
        value: The JSON object Value
        key: The field name
    
    Returns:
        The string value
    
    Example:
        var json = loads('{"name": "Alice"}')
        var name = get_string(json, "name")  # "Alice"
    """
    var raw_value = value.get(key)
    var parsed = loads[target="cpu"](raw_value)
    if not parsed.is_string():
        raise Error("Field '" + key + "' is not a string")
    return parsed.string_value()


fn get_int(value: Value, key: String) raises -> Int:
    """Extract an int field from a JSON object Value.
    
    Args:
        value: The JSON object Value
        key: The field name
    
    Returns:
        The int value
    
    Example:
        var json = loads('{"age": 30}')
        var age = get_int(json, "age")  # 30
    """
    var raw_value = value.get(key)
    var parsed = loads[target="cpu"](raw_value)
    if not parsed.is_int():
        raise Error("Field '" + key + "' is not an int")
    return Int(parsed.int_value())


fn get_bool(value: Value, key: String) raises -> Bool:
    """Extract a bool field from a JSON object Value.
    
    Args:
        value: The JSON object Value
        key: The field name
    
    Returns:
        The bool value
    
    Example:
        var json = loads('{"active": true}')
        var active = get_bool(json, "active")  # True
    """
    var raw_value = value.get(key)
    var parsed = loads[target="cpu"](raw_value)
    if not parsed.is_bool():
        raise Error("Field '" + key + "' is not a bool")
    return parsed.bool_value()


fn get_float(value: Value, key: String) raises -> Float64:
    """Extract a float field from a JSON object Value.
    
    Args:
        value: The JSON object Value
        key: The field name
    
    Returns:
        The float value
    
    Example:
        var json = loads('{"price": 19.99}')
        var price = get_float(json, "price")  # 19.99
    """
    var raw_value = value.get(key)
    var parsed = loads[target="cpu"](raw_value)
    if parsed.is_float():
        return parsed.float_value()
    elif parsed.is_int():
        return Float64(parsed.int_value())
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
            fn from_json(json: Value) raises -> Self:
                return Self(
                    name=get_string(json, "name"),
                    age=get_int(json, "age")
                )
        
        var person = deserialize[Person]('{"name":"Alice","age":30}')
    """
    @staticmethod
    fn from_json(json: Value) raises -> Self:
        """Deserialize from a JSON Value."""
        ...


fn deserialize[T: Deserializable, target: StaticString = "cpu"](
    json_str: String
) raises -> T:
    """Deserialize a JSON string into a typed object.
    
    The type must implement the Deserializable trait with a from_json() static method.
    
    Parameters:
        T: Type that implements Deserializable
        target: "cpu" (default) or "gpu" for parsing backend
    
    Args:
        json_str: JSON string to deserialize
    
    Returns:
        Deserialized object of type T
    
    Example:
        var person = deserialize[Person]('{"name":"Alice","age":30}')
        print(person.name)  # Alice
    """
    var json = loads[target](json_str)
    return T.from_json(json)


