# mojson - JSON serialization

from .value import Value


fn to_string(v: Value) -> String:
    """Convert a Value to a JSON string."""
    if v.is_null():
        return "null"
    elif v.is_bool():
        return "true" if v.bool_value() else "false"
    elif v.is_int():
        return String(v.int_value())
    elif v.is_float():
        return String(v.float_value())
    elif v.is_string():
        # Escape special characters
        var result = String('"')
        var s = v.string_value()
        for i in range(len(s)):
            var c = s[i]
            if c == '"':
                result += '\\"'
            elif c == "\\":
                result += "\\\\"
            elif c == "\n":
                result += "\\n"
            elif c == "\r":
                result += "\\r"
            elif c == "\t":
                result += "\\t"
            else:
                result += c
        result += '"'
        return result^
    elif v.is_array() or v.is_object():
        return v.raw_json()
    return "null"


fn dumps(v: Value) -> String:
    """Serialize a Value to JSON string (like Python's json.dumps).

    Args:
        v: Value to serialize

    Returns:
        JSON string representation

    Example:
        var data = loads('{"name": "Alice"}')
        print(dumps(data))  # {"name": "Alice"}
    """
    return to_string(v)


fn dump(v: Value, mut f: FileHandle) raises:
    """Serialize a Value and write to file (like Python's json.dump).

    Args:
        v: Value to serialize
        f: FileHandle to write JSON to

    Example:
        var data = loads('{"name": "Alice"}')
        with open("output.json", "w") as f:
            dump(data, f)
    """
    f.write(dumps(v))
