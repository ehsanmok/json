# mojson - JSON Parser
# Unified CPU/GPU parser with compile-time target selection

from collections import List
from memory import memcpy

from .value import Value, Null, make_array_value, make_object_value
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
        raise Error("Empty JSON")

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
    var result = String()
    var i = start + 1
    var escaped = False

    while i < len(data):
        var c = data[i]
        if escaped:
            if c == ord("n"):
                result += "\n"
            elif c == ord("t"):
                result += "\t"
            elif c == ord("r"):
                result += "\r"
            elif c == ord('"'):
                result += '"'
            elif c == ord("\\"):
                result += "\\"
            else:
                result += chr(Int(c))
            escaped = False
        elif c == ord("\\"):
            escaped = True
        elif c == 0x22:
            break
        else:
            result += chr(Int(c))
        i += 1

    return Value(result)


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
        for i in range(len(s)):
            if s[i] == "." or s[i] == "e" or s[i] == "E":
                is_float = True
                break
        if is_float:
            return Value(atof(s))
        return Value(atol(s))
    if c == 0x5B:
        return _build_array(iter, json)
    if c == 0x7B:
        return _build_object(iter, json)

    raise Error("Unknown JSON type")


fn _build_array(mut iter: JSONIterator, json: String) raises -> Value:
    """Build an array Value."""
    var raw = iter.get_value()
    var count = 0
    var depth = 0
    var in_string = False
    var escaped = False

    for i in range(len(raw)):
        var c = raw[i]
        if escaped:
            escaped = False
            continue
        if c == "\\":
            escaped = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "[" or c == "{":
            depth += 1
        elif c == "]" or c == "}":
            depth -= 1
        elif c == "," and depth == 1:
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

    for i in range(len(raw)):
        var c = raw[i]
        if escaped:
            escaped = False
            continue
        if c == "\\":
            escaped = True
            continue
        if c == '"':
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
        if c == "{" or c == "[":
            depth += 1
        elif c == "}" or c == "]":
            depth -= 1
        elif c == ":" and depth == 1:
            expect_key = False
        elif c == "," and depth == 1:
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
        var data = loads('{"name": "Alice"}')              # CPU (default)
        var data = loads[target="gpu"]('{"name": "Alice"}')  # GPU
    """

    @parameter
    if target == "cpu":
        return _parse_cpu(s)
    else:
        return _parse_gpu(s)


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
            var data = load(f)                    # CPU (default)
            var data = load[target="gpu"](f)      # GPU
    """
    var content = f.read()
    return loads[target](content)
