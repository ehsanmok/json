# mojson - JSON Parser
# Unified CPU/GPU parser with compile-time target selection

from collections import List
from memory import memcpy

from .value import Value, Null, make_array_value, make_object_value
from .serialize import dumps
from .cpu import SimdjsonFFI, SIMDJSON_TYPE_NULL, SIMDJSON_TYPE_BOOL
from .cpu import SIMDJSON_TYPE_INT64, SIMDJSON_TYPE_UINT64
from .cpu import SIMDJSON_TYPE_DOUBLE, SIMDJSON_TYPE_STRING
from .cpu import SIMDJSON_TYPE_ARRAY, SIMDJSON_TYPE_OBJECT
from .types import JSONInput, JSONResult
from .gpu import parse_json_gpu
from .iterator import JSONIterator


# =============================================================================
# CPU Parser (simdjson FFI)
# =============================================================================


fn _build_value_from_simdjson(
    ffi: SimdjsonFFI, value_handle: Int, raw_json: String
) raises -> Value:
    """Recursively build a Value tree from simdjson parse result."""
    var typ = ffi.get_type(value_handle)

    if typ == SIMDJSON_TYPE_NULL:
        return Value(Null())
    elif typ == SIMDJSON_TYPE_BOOL:
        return Value(ffi.get_bool(value_handle))
    elif typ == SIMDJSON_TYPE_INT64:
        return Value(ffi.get_int(value_handle))
    elif typ == SIMDJSON_TYPE_UINT64:
        return Value(Int64(ffi.get_uint(value_handle)))
    elif typ == SIMDJSON_TYPE_DOUBLE:
        return Value(ffi.get_float(value_handle))
    elif typ == SIMDJSON_TYPE_STRING:
        return Value(ffi.get_string(value_handle))
    elif typ == SIMDJSON_TYPE_ARRAY:
        var count = ffi.array_count(value_handle)
        return make_array_value(raw_json, count)
    elif typ == SIMDJSON_TYPE_OBJECT:
        var keys = List[String]()
        var iter = ffi.object_begin(value_handle)
        while not ffi.object_iter_done(iter):
            keys.append(ffi.object_iter_get_key(iter))
            ffi.object_iter_next(iter)
        ffi.object_iter_free(iter)
        return make_object_value(raw_json, keys^)
    else:
        raise Error("Unknown JSON value type")


fn _parse_cpu(s: String) raises -> Value:
    """Parse JSON using simdjson FFI."""
    var ffi = SimdjsonFFI()
    var root = ffi.parse(s)
    var result = _build_value_from_simdjson(ffi, root, s)
    ffi.free_value(root)
    ffi.destroy()
    return result^


# =============================================================================
# GPU Parser
# =============================================================================


fn _parse_gpu(s: String) raises -> Value:
    """Parse JSON using GPU."""
    var data = s.as_bytes()
    var start = 0

    # Skip leading whitespace
    while start < len(data) and (
        data[start] == 0x20
        or data[start] == 0x09
        or data[start] == 0x0A
        or data[start] == 0x0D
    ):
        start += 1

    if start >= len(data):
        from .errors import json_parse_error
        raise Error(json_parse_error("Empty or whitespace-only input", s, 0))

    var first_char = data[start]

    # Simple primitives - parse directly
    if first_char == ord("n"):
        return Value(Null())
    if first_char == ord("t"):
        return Value(True)
    if first_char == ord("f"):
        return Value(False)
    if first_char == 0x22:  # '"'
        return _parse_string_value(s, start)
    if first_char == ord("-") or (
        first_char >= ord("0") and first_char <= ord("9")
    ):
        return _parse_number_value(s, start)

    # Objects and arrays - use GPU parser
    # Create bytes once - used for both GPU parser and iterator
    var n = len(data)
    var bytes = List[UInt8](capacity=n)
    bytes.resize(n, 0)
    memcpy(dest=bytes.unsafe_ptr(), src=data.unsafe_ptr(), count=n)

    # GPU parser reads from bytes pointer, doesn't need ownership
    var input_obj = JSONInput(bytes.copy())  # Copy for GPU parser
    var result = parse_json_gpu(input_obj^)

    # Original bytes for iterator
    var iterator = JSONIterator(result^, bytes^)

    return _build_value(iterator, s)


fn _parse_string_value(s: String, start: Int) raises -> Value:
    """Parse a string value."""
    var data = s.as_bytes()
    var n = len(data)
    var i = start + 1
    
    # Find end of string
    var end_idx = i
    var has_escapes = False
    while end_idx < n:
        var c = data[end_idx]
        if c == ord("\\"):
            has_escapes = True
            end_idx += 2
            continue
        if c == 0x22:  # "
            break
        end_idx += 1
    
    # Fast path: no escapes
    if not has_escapes:
        return Value(String(s[i:end_idx]))
    
    # Slow path: handle escapes including \uXXXX
    from .unicode import unescape_json_string
    var bytes_list = List[UInt8](capacity=n)
    for j in range(n):
        bytes_list.append(data[j])
    var unescaped = unescape_json_string(bytes_list, i, end_idx)
    return Value(String(unsafe_from_utf8=unescaped^))


fn _parse_number_value(s: String, start: Int) raises -> Value:
    """Parse a number value."""
    var data = s.as_bytes()
    var num_str = String()
    var is_float = False
    var i = start

    while i < len(data):
        var c = data[i]
        if c == ord("-") or c == ord("+") or (c >= ord("0") and c <= ord("9")):
            num_str += chr(Int(c))
        elif c == ord(".") or c == ord("e") or c == ord("E"):
            num_str += chr(Int(c))
            is_float = True
        else:
            break
        i += 1

    if is_float:
        return Value(atof(num_str))
    else:
        return Value(atol(num_str))


fn _build_value(mut iter: JSONIterator, json: String) raises -> Value:
    """Build a Value tree from JSONIterator."""
    var c = iter.get_current_char()

    if c == ord("n"):
        return Value(Null())
    if c == ord("t"):
        return Value(True)
    if c == ord("f"):
        return Value(False)
    if c == 0x22:
        return Value(iter.get_value())
    if c == ord("-") or (c >= ord("0") and c <= ord("9")):
        var s = iter.get_value()
        var is_float = False
        var s_bytes = s.as_bytes()
        for i in range(len(s_bytes)):
            var ch = s_bytes[i]
            if ch == ord(".") or ch == ord("e") or ch == ord("E"):
                is_float = True
                break
        if is_float:
            return Value(atof(s))
        return Value(atol(s))
    if c == 0x5B:
        return _build_array(iter, json)
    if c == 0x7B:
        return _build_object(iter, json)

    from .errors import json_parse_error
    var pos = iter.get_position()
    raise Error(json_parse_error("Unexpected character", json, pos))


fn _build_array(mut iter: JSONIterator, json: String) raises -> Value:
    """Build an array Value."""
    var raw = iter.get_value()
    var raw_bytes = raw.as_bytes()
    var count = 0
    var depth = 0
    var in_string = False
    var escaped = False

    for i in range(len(raw_bytes)):
        var c = raw_bytes[i]
        if escaped:
            escaped = False
            continue
        if c == ord("\\"):
            escaped = True
            continue
        if c == ord('"'):
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == ord("[") or c == ord("{"):
            depth += 1
        elif c == ord("]") or c == ord("}"):
            depth -= 1
        elif c == ord(",") and depth == 1:
            count += 1

    if len(raw) > 2:
        count += 1

    return make_array_value(raw, count)


fn _build_object(mut iter: JSONIterator, json: String) raises -> Value:
    """Build an object Value."""
    var raw = iter.get_value()
    var raw_bytes = raw.as_bytes()
    var keys = List[String]()
    var depth = 0
    var in_string = False
    var escaped = False
    var key_start = -1
    var expect_key = True

    for i in range(len(raw_bytes)):
        var c = raw_bytes[i]
        if escaped:
            escaped = False
            continue
        if c == ord("\\"):
            escaped = True
            continue
        if c == ord('"'):
            if not in_string:
                in_string = True
                if depth == 1 and expect_key:
                    key_start = i + 1
            else:
                in_string = False
                if key_start >= 0 and depth == 1:
                    var key_len = i - key_start
                    var key_bytes = List[UInt8](capacity=key_len)
                    key_bytes.resize(key_len, 0)
                    memcpy(
                        dest=key_bytes.unsafe_ptr(),
                        src=raw_bytes.unsafe_ptr() + key_start,
                        count=key_len,
                    )
                    keys.append(String(unsafe_from_utf8=key_bytes^))
                    key_start = -1
            continue
        if in_string:
            continue
        if c == ord("{") or c == ord("["):
            depth += 1
        elif c == ord("}") or c == ord("]"):
            depth -= 1
        elif c == ord(":") and depth == 1:
            expect_key = False
        elif c == ord(",") and depth == 1:
            expect_key = True

    return make_object_value(raw, keys^)


# =============================================================================
# Public API (Python-compatible)
# =============================================================================


fn loads[target: StaticString = "cpu"](s: String) raises -> Value:
    """Deserialize JSON string to a Value (like Python's json.loads).

    Parameters:
        target: "cpu" (default) or "gpu"

    Args:
        s: JSON string to parse

    Returns:
        Parsed Value

    Example:
        var data = loads('{"name": "Alice"}')
        var data = loads[target="gpu"](large_json)  # GPU for large files
    """
    @parameter
    if target == "cpu":
        return _parse_cpu(s)
    else:
        return _parse_gpu(s)


fn loads[target: StaticString = "cpu"](s: String, config: ParserConfig) raises -> Value:
    """Deserialize JSON with custom configuration.
    
    Parameters:
        target: "cpu" (default) or "gpu"
    
    Args:
        s: JSON string to parse
        config: Parser configuration (allow_comments, allow_trailing_comma, max_depth)
    
    Returns:
        Parsed Value
    
    Example:
        var data = loads('{"a": 1} // comment', ParserConfig(allow_comments=True))
    """
    from .config import preprocess_json
    var preprocessed = preprocess_json(s, config)
    return loads[target](preprocessed)


fn loads[
    target: StaticString = "cpu",
    format: StaticString = "json",
](s: String) raises -> List[Value]:
    """Deserialize NDJSON string to a list of Values.

    Parameters:
        target: "cpu" (default) or "gpu"
        format: Must be "ndjson" for this overload

    Args:
        s: NDJSON string (one JSON value per line)

    Returns:
        List of parsed Values

    Example:
        var values = loads[format="ndjson"]('{"a":1}\\n{"a":2}')
    """
    @parameter
    if format != "ndjson":
        constrained[False, "Use format='ndjson' for List[Value] return type"]()
    
    from .ndjson import _split_lines, _is_whitespace_only
    var result = List[Value]()
    var lines = _split_lines(s)
    
    for i in range(len(lines)):
        var line = lines[i]
        if _is_whitespace_only(line):
            continue
        var value = loads[target](line)
        result.append(value^)
    
    return result^


fn loads[lazy: Bool](s: String) raises -> LazyValue:
    """Create a lazy JSON value for on-demand parsing (CPU only).

    Parameters:
        lazy: Must be True (required, no default)

    Args:
        s: JSON string

    Returns:
        LazyValue that parses on demand

    Example:
        var lazy = loads[lazy=True](huge_json)
        var name = lazy.get("/users/0/name")  # Only parses this path
    
    Note:
        Lazy parsing is CPU-only. For GPU, use loads[target="gpu"] directly.
    """
    @parameter
    if not lazy:
        constrained[False, "Use lazy=True for LazyValue return type"]()
    
    return LazyValue(s)


fn load[target: StaticString = "cpu"](mut f: FileHandle) raises -> Value:
    """Deserialize JSON from file to a Value (like Python's json.load).

    Parameters:
        target: "cpu" (default) or "gpu"

    Args:
        f: FileHandle to read JSON from

    Returns:
        Parsed Value

    Example:
        with open("data.json", "r") as f:
            var data = load(f)
    """
    var content = f.read()
    return loads[target](content)


fn load[target: StaticString = "cpu"](mut f: FileHandle, config: ParserConfig) raises -> Value:
    """Deserialize JSON from file with custom configuration.

    Parameters:
        target: "cpu" (default) or "gpu"

    Args:
        f: FileHandle to read JSON from
        config: Parser configuration

    Returns:
        Parsed Value
    """
    var content = f.read()
    return loads[target](content, config)


fn load[target: StaticString = "cpu"](path: String) raises -> Value:
    """Load JSON/NDJSON from file path. Auto-detects format from extension.

    Parameters:
        target: "cpu" (default) or "gpu"

    Args:
        path: Path to .json or .ndjson file

    Returns:
        Value (for .json) or Value array (for .ndjson)

    Example:
        var data = load("config.json")           # Returns object/value
        var items = load("data.ndjson")          # Returns array of values
        var big = load[target="gpu"]("large.json")
    """
    var f = open(path, "r")
    var content = f.read()
    f.close()
    
    # Auto-detect NDJSON from extension
    if path.endswith(".ndjson"):
        var values = loads[target, format="ndjson"](content)
        return _list_to_array_value(values)
    
    return loads[target](content)


fn _list_to_array_value(values: List[Value]) -> Value:
    """Convert List[Value] to a Value containing an array."""
    var count = len(values)
    if count == 0:
        return make_array_value("[]", 0)
    
    var raw = String("[")
    for i in range(count):
        if i > 0:
            raw += ","
        raw += dumps(values[i])
    raw += "]"
    return make_array_value(raw, count)


fn load[streaming: Bool](path: String) raises -> StreamingParser:
    """Stream large files line by line (CPU only, for memory efficiency).

    Parameters:
        streaming: Must be True

    Args:
        path: Path to NDJSON file

    Returns:
        StreamingParser iterator

    Example:
        var parser = load[streaming=True]("huge.ndjson")
        while parser.has_next():
            var item = parser.next()
        parser.close()
    
    Note:
        Streaming is CPU-only (for memory efficiency, not speed).
        For GPU speed on files that fit in memory, use:
        load[target="gpu"]("file.ndjson")
    """
    @parameter
    if not streaming:
        constrained[False, "Use streaming=True for StreamingParser"]()
    
    return StreamingParser(path)


# Backwards compatibility aliases (deprecated, use loads/load instead)
fn loads_with_config[target: StaticString = "cpu"](
    s: String, config: ParserConfig
) raises -> Value:
    """Deprecated: Use loads(s, config) instead."""
    return loads[target](s, config)


from .config import ParserConfig
from .lazy import LazyValue
from .streaming import StreamingParser
