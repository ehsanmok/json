# json - JSON serialization

from .config import SerializerConfig
from .value import Value
from .value.raw_ops import escape_json_string


def _escape_string(s: String) -> String:
    """Quote and escape `s` as a JSON string literal.

    Delegates to the one canonical implementation; see
    `json/value/raw_ops.mojo`. This copy previously lacked the 0x08 and
    0x0C short forms, emitting them as six-character hex escapes -- legal
    JSON, but different bytes from the other paths for the same input.
    """
    return escape_json_string(s)


def to_string(v: Value) -> String:
    """Convert a Value to a JSON string (compact)."""
    if v.is_null():
        return "null"
    elif v.is_bool():
        return "true" if v.bool_value() else "false"
    elif v.is_int():
        return String(v.int_value())
    elif v.is_uint():
        return String(v.uint_value())
    elif v.is_float():
        return String(v.float_value())
    elif v.is_string():
        return _escape_string(v.string_value())
    elif v.is_array() or v.is_object():
        return v.raw_json()
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
    return v.pretty_json(indent)


def dumps(v: Value, config: SerializerConfig) -> String:
    """Serialize a Value with custom configuration.

    Every option is honoured by the writer during the one structural
    walk. They used to be applied afterwards by re-scanning the
    finished text: `sort_keys` was a function that returned its input
    unchanged, and the two escaping options rebuilt the string through
    `chr` per byte, so any character above U+007F came out as the
    Latin-1 reading of its UTF-8 bytes.

    Args:
        v: Value to serialize.
        config: Serializer configuration.

    Returns:
        JSON string representation.

    Example:
        var json = dumps(value, SerializerConfig(indent="  ", escape_unicode=True)).
    """
    return v.to_json(
        indent=config.indent,
        ascii_only=config.escape_unicode,
        escape_solidus=config.escape_forward_slash,
        sort_keys=config.sort_keys,
    )


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


def dumps[
    format: StaticString = "json"
](values: List[Value], config: SerializerConfig) -> String:
    """Serialize a list of Values to NDJSON with custom configuration.

    The indent option is ignored here: NDJSON puts one value per line,
    so a value spread over several lines would not be readable back.

    Parameters:
        format: Must be "ndjson" for this overload.

    Args:
        values: List of Values to serialize.
        config: Serializer configuration.

    Returns:
        NDJSON string (one JSON value per line).
    """

    comptime if format != "ndjson":
        comptime assert False, "Use format='ndjson' for List[Value] input"

    var result = String()
    for i in range(len(values)):
        if i > 0:
            result += "\n"
        result += values[i].to_json(
            ascii_only=config.escape_unicode,
            escape_solidus=config.escape_forward_slash,
            sort_keys=config.sort_keys,
        )
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


def dump(v: Value, mut f: FileHandle, config: SerializerConfig) raises:
    """Serialize a Value with custom configuration and write to file.

    Args:
        v: Value to serialize.
        f: FileHandle to write JSON to.
        config: Serializer configuration.

    Example:
        with open("output.json", "w") as f:
            dump(data, f, SerializerConfig(indent="  ", sort_keys=True)).
    """
    f.write(dumps(v, config))


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
