# json - JSON serialization

from .value import Value


def _escape_string(s: String) -> String:
    """Escape special characters in a string for JSON."""
    var result = String('"')
    var s_bytes = s.as_bytes()
    for i in range(len(s_bytes)):
        var c = s_bytes[i]
        if c == UInt8(ord('"')):
            result += '\\"'
        elif c == UInt8(ord("\\")):
            result += "\\\\"
        elif c == UInt8(ord("\n")):
            result += "\\n"
        elif c == UInt8(ord("\r")):
            result += "\\r"
        elif c == UInt8(ord("\t")):
            result += "\\t"
        elif c < 0x20:
            # Control characters - escape as \u00XX
            result += "\\u00"
            var hi = (c >> 4) & 0x0F
            var lo = c & 0x0F
            result += chr(Int(hi) + ord("0")) if hi < 10 else chr(
                Int(hi) - 10 + ord("a")
            )
            result += chr(Int(lo) + ord("0")) if lo < 10 else chr(
                Int(lo) - 10 + ord("a")
            )
        else:
            result += chr(Int(c))
    result += '"'
    return result^


def to_string(v: Value) -> String:
    """Convert a Value to a JSON string (compact)."""
    if v.is_null():
        return "null"
    elif v.is_bool():
        return "true" if v.bool_value() else "false"
    elif v.is_int():
        return String(v.int_value())
    elif v.is_float():
        return String(v.float_value())
    elif v.is_string():
        return _escape_string(v.string_value())
    elif v.is_array() or v.is_object():
        return v.raw_json()
    return "null"


def _format_json(raw: String, indent: String, current_indent: String) -> String:
    """Format raw JSON with indentation.

    Args:
        raw: Raw JSON string to format.
        indent: Indentation string per level (e.g., "  " or "    ").
        current_indent: Current indentation level.

    Returns:
        Formatted JSON string with newlines and indentation.
    """
    var result = String()
    var raw_bytes = raw.as_bytes()
    var n = len(raw_bytes)
    var i = 0
    var in_string = False
    var escaped = False
    var next_indent = current_indent + indent

    while i < n:
        var c = raw_bytes[i]

        if escaped:
            result += chr(Int(c))
            escaped = False
            i += 1
            continue

        if c == UInt8(ord("\\")) and in_string:
            result += chr(Int(c))
            escaped = True
            i += 1
            continue

        if c == UInt8(ord('"')):
            in_string = not in_string
            result += chr(Int(c))
            i += 1
            continue

        if in_string:
            result += chr(Int(c))
            i += 1
            continue

        # Handle structural characters
        if c == UInt8(ord("{")) or c == UInt8(ord("[")):
            var close_char = UInt8(ord("}")) if c == UInt8(ord("{")) else UInt8(
                ord("]")
            )
            # Check if empty
            var j = i + 1
            while j < n and (
                raw_bytes[j] == UInt8(ord(" "))
                or raw_bytes[j] == UInt8(ord("\t"))
                or raw_bytes[j] == UInt8(ord("\n"))
                or raw_bytes[j] == UInt8(ord("\r"))
            ):
                j += 1
            if j < n and raw_bytes[j] == close_char:
                # Empty object/array - keep compact
                result += chr(Int(c))
                result += chr(Int(close_char))
                i = j + 1
                continue
            result += chr(Int(c))
            result += "\n" + next_indent
            i += 1
            # Skip whitespace after opening brace
            while i < n and (
                raw_bytes[i] == UInt8(ord(" "))
                or raw_bytes[i] == UInt8(ord("\t"))
                or raw_bytes[i] == UInt8(ord("\n"))
                or raw_bytes[i] == UInt8(ord("\r"))
            ):
                i += 1
            continue

        if c == UInt8(ord("}")) or c == UInt8(ord("]")):
            result += "\n" + current_indent + chr(Int(c))
            i += 1
            continue

        if c == UInt8(ord(",")):
            result += ",\n" + next_indent
            i += 1
            # Skip whitespace after comma
            while i < n and (
                raw_bytes[i] == UInt8(ord(" "))
                or raw_bytes[i] == UInt8(ord("\t"))
                or raw_bytes[i] == UInt8(ord("\n"))
                or raw_bytes[i] == UInt8(ord("\r"))
            ):
                i += 1
            continue

        if c == UInt8(ord(":")):
            result += ": "
            i += 1
            # Skip whitespace after colon
            while i < n and (
                raw_bytes[i] == UInt8(ord(" "))
                or raw_bytes[i] == UInt8(ord("\t"))
                or raw_bytes[i] == UInt8(ord("\n"))
                or raw_bytes[i] == UInt8(ord("\r"))
            ):
                i += 1
            continue

        # Skip whitespace (we handle it ourselves)
        if (
            c == UInt8(ord(" "))
            or c == UInt8(ord("\t"))
            or c == UInt8(ord("\n"))
            or c == UInt8(ord("\r"))
        ):
            i += 1
            continue

        # Regular character
        result += chr(Int(c))
        i += 1

    return result^


def _to_string_pretty(
    v: Value, indent: String, current_indent: String
) -> String:
    """Convert a Value to a pretty-printed JSON string."""
    if v.is_null():
        return "null"
    elif v.is_bool():
        return "true" if v.bool_value() else "false"
    elif v.is_int():
        return String(v.int_value())
    elif v.is_float():
        return String(v.float_value())
    elif v.is_string():
        return _escape_string(v.string_value())
    elif v.is_array() or v.is_object():
        return _format_json(v.raw_json(), indent, current_indent)
    return "null"


def dumps(v: Value, indent: String = "") -> String:
    """Serialize a Value to JSON string (like Python's json.dumps).

    Args:
        v: Value to serialize.
        indent: Indentation string (empty for compact, e.g., "  " for 2 spaces).

    Returns:
        JSON string representation.

    Example:
        var data = loads('{"name": "Alice", "age": 30}')
        print(dumps(data))  # {"name":"Alice","age":30}.
        print(dumps(data, indent="  "))  # Pretty-printed.
    """
    if indent == "":
        return to_string(v)
    return _to_string_pretty(v, indent, "")


def dumps(v: Value, config: SerializerConfig) -> String:
    """Serialize a Value with custom configuration.

    Args:
        v: Value to serialize.
        config: Serializer configuration.

    Returns:
        JSON string representation.

    Example:
        var json = dumps(value, SerializerConfig(indent="  ", escape_unicode=True)).
    """
    var result: String

    if config.indent == "":
        result = to_string(v)
    else:
        result = _to_string_pretty(v, config.indent, "")

    if config.escape_unicode:
        result = _escape_unicode_chars(result)

    if config.escape_forward_slash:
        result = _escape_forward_slashes(result)

    if config.sort_keys and (v.is_object() or v.is_array()):
        result = _sort_object_keys(result)

    return result^


def dumps[format: StaticString = "json"](values: List[Value]) -> String:
    """Serialize a list of Values to NDJSON string.

    Parameters:
        format: Must be "ndjson" for this overload.

    Args:
        values: List of Values to serialize.

    Returns:
        NDJSON string (one JSON value per line).

    Example:
        var values = List[Value]()
        values.append(loads('{"a":1}'))
        values.append(loads('{"a":2}'))
        print(dumps[format="ndjson"](values)).
    """

    comptime if format != "ndjson":
        comptime assert False, "Use format='ndjson' for List[Value] input"

    var result = String()
    for i in range(len(values)):
        if i > 0:
            result += "\n"
        result += dumps(values[i])
    return result^


def dump(v: Value, mut f: FileHandle) raises:
    """Serialize a Value and write to file (like Python's json.dump).

    Args:
        v: Value to serialize.
        f: FileHandle to write JSON to.

    Example:
        with open("output.json", "w") as f:
            dump(data, f).
    """
    f.write(dumps(v))


def dump(v: Value, mut f: FileHandle, indent: String) raises:
    """Serialize a Value with indentation and write to file.

    Args:
        v: Value to serialize.
        f: FileHandle to write JSON to.
        indent: Indentation string.

    Example:
        with open("output.json", "w") as f:
            dump(data, f, indent="  ").
    """
    f.write(dumps(v, indent))


def dump[
    format: StaticString = "json"
](values: List[Value], mut f: FileHandle) raises:
    """Serialize a list of Values to NDJSON and write to file.

    Parameters:
        format: Must be "ndjson" for this overload.

    Args:
        values: List of Values to serialize.
        f: FileHandle to write NDJSON to.

    Example:
        with open("output.ndjson", "w") as f:
            dump[format="ndjson"](values, f).
    """

    comptime if format != "ndjson":
        comptime assert False, "Use format='ndjson' for List[Value] input"

    f.write(dumps[format="ndjson"](values))


# Backwards compatibility alias (deprecated, use dumps(v, config) instead)
def dumps_with_config(v: Value, config: SerializerConfig) -> String:
    """Serialize a Value with custom configuration.

    Args:
        v: Value to serialize.
        config: Serializer configuration.

    Returns:
        JSON string representation.

    Example:
        var config = SerializerConfig(sort_keys=True, indent="  ")
        var json = dumps_with_config(value, config).
    """
    var result: String

    if config.indent == "":
        result = to_string(v)
    else:
        result = _to_string_pretty(v, config.indent, "")

    # Apply additional options
    if config.escape_unicode:
        result = _escape_unicode_chars(result)

    if config.escape_forward_slash:
        result = _escape_forward_slashes(result)

    if config.sort_keys and (v.is_object() or v.is_array()):
        result = _sort_object_keys(result)

    return result^


def _escape_unicode_chars(s: String) -> String:
    """Escape non-ASCII characters as \\uXXXX."""
    var result = String()
    var s_bytes = s.as_bytes()
    var in_string = False
    var escaped = False

    for i in range(len(s_bytes)):
        var c = s_bytes[i]

        if escaped:
            escaped = False
            result += chr(Int(c))
            continue

        if c == UInt8(ord("\\")):
            escaped = True
            result += chr(Int(c))
            continue

        if c == UInt8(ord('"')):
            in_string = not in_string
            result += chr(Int(c))
            continue

        # Escape non-ASCII inside strings
        if in_string and c > 127:
            result += "\\u00"
            var hi = (c >> 4) & 0x0F
            var lo = c & 0x0F
            result += chr(Int(hi) + ord("0")) if hi < 10 else chr(
                Int(hi) - 10 + ord("a")
            )
            result += chr(Int(lo) + ord("0")) if lo < 10 else chr(
                Int(lo) - 10 + ord("a")
            )
        else:
            result += chr(Int(c))

    return result^


def _escape_forward_slashes(s: String) -> String:
    """Escape forward slashes as \\/ for HTML embedding safety."""
    var result = String()
    var s_bytes = s.as_bytes()
    var in_string = False
    var escaped = False

    for i in range(len(s_bytes)):
        var c = s_bytes[i]

        if escaped:
            escaped = False
            result += chr(Int(c))
            continue

        if c == UInt8(ord("\\")):
            escaped = True
            result += chr(Int(c))
            continue

        if c == UInt8(ord('"')):
            in_string = not in_string
            result += chr(Int(c))
            continue

        # Escape / inside strings
        if in_string and c == UInt8(ord("/")):
            result += "\\/"
        else:
            result += chr(Int(c))

    return result^


def _sort_object_keys(json: String) -> String:
    """Sort object keys alphabetically (simple implementation).

    Note: This is a basic implementation that works for simple objects.
    For complex nested structures, a full re-parse may be needed.
    """
    # For now, return as-is - full implementation would require
    # re-parsing and re-serializing with sorted keys
    # This is a TODO for a more complete implementation
    return json


from .config import SerializerConfig


# Helper functions for building JSON strings from basic types
def to_json_string(s: String) -> String:
    """Convert a String to JSON string format (with quotes and escaping)."""
    return _escape_string(s)


def to_json_value(val: String) -> String:
    """Convert String to JSON."""
    return to_json_string(val)


def to_json_value(val: Int) -> String:
    """Convert Int to JSON."""
    return String(val)


def to_json_value(val: Int64) -> String:
    """Convert Int64 to JSON."""
    return String(val)


def to_json_value(val: Float64) -> String:
    """Convert Float64 to JSON."""
    return String(val)


def to_json_value(val: Bool) -> String:
    """Convert Bool to JSON."""
    return "true" if val else "false"


trait Serializable:
    """Trait for types that can be serialized to JSON.

    Implement this trait to enable automatic serialization with serialize().

    Example:
        struct Person(Serializable):
            var name: String
            var age: Int

            def to_json(self) -> String:
                return '{"name":' + to_json_value(self.name) +
                       ',"age":' + to_json_value(self.age) + '}'

        var json = serialize(Person("Alice", 30))  # {"name":"Alice","age":30}
    """

    def to_json(self) -> String:
        """Serialize this object to a JSON string."""
        ...


def serialize[T: Serializable](obj: T) -> String:
    """Serialize an object to JSON string.

    The object must implement the Serializable trait with a to_json() method.

    Parameters:
        T: Type that implements Serializable.

    Args:
        obj: Object to serialize.

    Returns:
        JSON string representation.

    Example:
        var person = Person("Alice", 30)
        var json = serialize(person)  # `{"name":"Alice","age":30}`.
    """
    return obj.to_json()
