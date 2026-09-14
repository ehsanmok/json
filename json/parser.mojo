# json - JSON Parser
# Unified CPU/GPU parser with compile-time target and backend selection

from std.collections import List
from std.memory import unsafe_memcpy, ArcPointer

from .value import Value, Null, make_view_value
from .serialize import dumps
from .cpu import SimdjsonFFI, SIMDJSON_TYPE_NULL, SIMDJSON_TYPE_BOOL
from .cpu import SIMDJSON_TYPE_INT64, SIMDJSON_TYPE_UINT64
from .cpu import SIMDJSON_TYPE_DOUBLE, SIMDJSON_TYPE_STRING
from .cpu import SIMDJSON_TYPE_ARRAY, SIMDJSON_TYPE_OBJECT
from .cpu import parse_cpu_native_tape
from .cpu.validate import is_valid_utf8
from .document import (
    Document,
    pack_tape_entry,
    pack_pair,
    TAPE_TAG_NULL,
    TAPE_TAG_BOOL,
    TAPE_TAG_INT,
    TAPE_TAG_FLOAT,
    TAPE_TAG_STRING_OWNED,
    TAPE_TAG_KEY,
    TAPE_TAG_ARRAY,
    TAPE_TAG_OBJECT,
)
from .types import JSONInput, JSONResult


# =============================================================================
# CPU Parser (simdjson FFI)
# =============================================================================


def _emit_simdjson_value(
    ffi: SimdjsonFFI, mut doc: Document, value_handle: Int
) raises -> UInt64:
    """Walk a simdjson value handle and translate it into tape entries
    in `doc`, returning the packed header for THIS value. The header
    is NOT appended to `doc.tape` -- the caller (a container above us
    or `_parse_cpu_simdjson` for the root) is responsible for placing
    it in the tape's parent header run.

    For containers we recursively flush our descendants' headers in
    one contiguous run before returning, matching the layout pinned in
    `json/document.mojo` and produced by `_emit_value_to_doc` in
    `json/cpu/stage2.mojo`:

    * ARRAY children are a flat run of `count` value headers.
    * OBJECT children alternate KEY / VALUE / KEY / VALUE ... so a
      `pair_count` of N occupies `2 * N` slots.

    All strings (scalar values and object keys) come back from the FFI
    as fully decoded `String`s, so they're interned into
    `string_pool` / `key_pool` and referenced with STRING_OWNED / KEY
    tags rather than the zero-copy `(offset, length)` slice form.
    """
    var typ = ffi.get_type(value_handle)
    var payload_mask = (UInt64(1) << 60) - 1

    if typ == SIMDJSON_TYPE_NULL:
        return pack_tape_entry(TAPE_TAG_NULL, 0)
    elif typ == SIMDJSON_TYPE_BOOL:
        var b: UInt64 = 1 if ffi.get_bool(value_handle) else 0
        return pack_tape_entry(TAPE_TAG_BOOL, b)
    elif typ == SIMDJSON_TYPE_INT64:
        var v = ffi.get_int(value_handle)
        return pack_tape_entry(TAPE_TAG_INT, UInt64(v) & payload_mask)
    elif typ == SIMDJSON_TYPE_UINT64:
        # Existing FFI API exposes uints as Int64 to callers; preserve
        # that shape so backend equivalence holds.
        var v = Int64(ffi.get_uint(value_handle))
        return pack_tape_entry(TAPE_TAG_INT, UInt64(v) & payload_mask)
    elif typ == SIMDJSON_TYPE_DOUBLE:
        var pool_idx = len(doc.float_pool)
        doc.float_pool.append(ffi.get_float(value_handle))
        return pack_tape_entry(TAPE_TAG_FLOAT, UInt64(pool_idx))
    elif typ == SIMDJSON_TYPE_STRING:
        var s = ffi.get_string(value_handle)
        var pool_idx = len(doc.string_pool)
        doc.string_pool.append(s^)
        return pack_tape_entry(TAPE_TAG_STRING_OWNED, UInt64(pool_idx))
    elif typ == SIMDJSON_TYPE_ARRAY:
        var count = ffi.array_count(value_handle)
        var iter = ffi.array_begin(value_handle)
        var headers = List[UInt64](capacity=count)
        while not ffi.array_iter_done(iter):
            var child_handle = ffi.array_iter_get(iter)
            headers.append(_emit_simdjson_value(ffi, doc, child_handle))
            ffi.array_iter_next(iter)
        ffi.array_iter_free(iter)
        var child_start = len(doc.tape)
        for j in range(len(headers)):
            doc.tape.append(headers[j])
        return pack_tape_entry(
            TAPE_TAG_ARRAY,
            pack_pair(UInt64(count), UInt64(child_start)),
        )
    elif typ == SIMDJSON_TYPE_OBJECT:
        var pair_count = ffi.object_count(value_handle)
        var iter = ffi.object_begin(value_handle)
        var headers = List[UInt64](capacity=2 * pair_count)
        while not ffi.object_iter_done(iter):
            var key = ffi.object_iter_get_key(iter)
            var key_pool_idx = len(doc.key_pool)
            doc.key_pool.append(key^)
            var key_header = pack_tape_entry(TAPE_TAG_KEY, UInt64(key_pool_idx))
            var value_handle_child = ffi.object_iter_get_value(iter)
            var value_header = _emit_simdjson_value(
                ffi, doc, value_handle_child
            )
            headers.append(key_header)
            headers.append(value_header)
            ffi.object_iter_next(iter)
        ffi.object_iter_free(iter)
        var child_start = len(doc.tape)
        for j in range(len(headers)):
            doc.tape.append(headers[j])
        return pack_tape_entry(
            TAPE_TAG_OBJECT,
            pack_pair(UInt64(pair_count), UInt64(child_start)),
        )
    else:
        raise Error("Unknown JSON value type")


def _parse_cpu_simdjson(s: String) raises -> Value:
    """Parse JSON using simdjson FFI backend, returning a tape-backed view.

    The simdjson DOM is walked once and translated into a `Document`.
    The root header is appended last so `Document.root()`'s
    "last entry is the root" invariant holds. The returned `Value`
    is a view over that document; callers see the same shape and
    accessor behaviour as the native Mojo tape parser
    (`parse_cpu_native_tape`).
    """
    var ffi = SimdjsonFFI()
    var root = ffi.parse(s)
    var doc = Document()
    var root_header = _emit_simdjson_value(ffi, doc, root)
    doc.tape.append(root_header)
    var root_idx = doc.root()
    ffi.free_value(root)
    ffi.destroy()
    var arc = ArcPointer[Document](doc^)
    return make_view_value(arc, root_idx)


def _parse_cpu_mojo(var s: String) raises -> Value:
    """Parse JSON using the two-pass CPU parser (stage 1 + stage 2)
    into a tape-backed `Document`. The returned `Value` is a view over
    that document.

    The stage 1 default is SIMD (1.5x to 2.2x faster than the scalar
    walker on the benchmark corpora). Differential testing routes
    through `cpu.parse_cpu_native_tape[force_scalar=True]`.
    """
    return parse_cpu_native_tape(s^)


def _parse_cpu[backend: StaticString = "simdjson"](s: String) raises -> Value:
    """Parse JSON using specified CPU backend.

    Parameters:
        backend: "simdjson" (default, FFI) or "mojo" (two-pass native
            parser).

    Args:
        s: JSON string to parse.

    Returns:
        Parsed Value.
    """

    comptime if backend == "simdjson":
        return _parse_cpu_simdjson(s)
    elif backend == "mojo":
        return _parse_cpu_mojo(s)
    else:
        comptime assert False, "Unknown backend: use 'simdjson' or 'mojo'"


def parse_string_scalar(s: String, start: Int) raises -> Value:
    """Parse a string value."""
    var data = s.as_bytes()
    var n = len(data)
    var i = start + 1

    # Find end of string
    var end_idx = i
    var has_escapes = False
    while end_idx < n:
        var c = data[end_idx]
        if c == UInt8(ord("\\")):
            has_escapes = True
            end_idx += 2
            continue
        if c == 0x22:  # "
            break
        end_idx += 1

    # Fast path: no escapes
    if not has_escapes:
        return Value(String(String(unsafe_from_utf8=s.as_bytes()[i:end_idx])))

    # Slow path: handle escapes including \uXXXX. Span-based so the
    # entire input doesn't get copied into a List[UInt8] per string.
    var unescaped = unescape_json_string_span(data, i, end_idx)
    return Value(String(unsafe_from_utf8=unescaped^))


def parse_number_scalar(s: String, start: Int) raises -> Value:
    """Parse a number value."""
    var data = s.as_bytes()
    var num_str = String()
    var is_float = False
    var i = start

    while i < len(data):
        var c = data[i]
        if (
            c == UInt8(ord("-"))
            or c == UInt8(ord("+"))
            or (c >= UInt8(ord("0")) and c <= UInt8(ord("9")))
        ):
            num_str += chr(Int(c))
        elif (
            c == UInt8(ord(".")) or c == UInt8(ord("e")) or c == UInt8(ord("E"))
        ):
            num_str += chr(Int(c))
            is_float = True
        else:
            break
        i += 1

    if is_float:
        return Value(atof(num_str))
    else:
        return Value(atol(num_str))


# =============================================================================
# Public API (Python-compatible)
# =============================================================================


def loads[target: StaticString = "cpu"](var s: String) raises -> Value:
    """Deserialize JSON string to a Value (like Python's json.loads).

    The parsed document owns its input, so `s` is taken by value. Pass
    a string you no longer need with `^` and it is moved rather than
    copied.

    Parameters:
        target: Parsing target/backend. Options: "cpu" (default, pure Mojo),
            "cpu-simdjson" (FFI), or "gpu" (for large files).

    Args:
        s: JSON string to parse.

    Returns:
        Parsed Value.

    Example:
        var data = loads('{"name": "Alice"}')
        var data = loads[target="gpu"](large_json)  # GPU for large files.
        var data = loads[target="cpu-simdjson"](s)  # Use simdjson FFI.
    """

    return _loads_value[target](s^)


def _loads_value[target: StaticString](var s: String) raises -> Value:
    """The body of `loads`, callable without overload ambiguity.

    `loads` is overloaded on its parameters as well as its arguments,
    so a call with one explicit parameter inside this module can match
    the NDJSON form. Calls that mean "one value" come here instead.
    """
    comptime if target == "cpu":
        return _parse_cpu_mojo(s^)
    elif target == "cpu-simdjson":
        return _parse_cpu["simdjson"](s)
    elif target == "gpu":
        # The GPU pipeline cannot be reached from here. Mojo resolves
        # every import statement it can see, whether or not the branch
        # holding it survives `comptime if`, so a `from .gpu import ...`
        # anywhere in this module would make `max-core` a hard
        # requirement of `import json`. Keeping the GPU entry point in
        # `json/gpu/backend.mojo`, which nothing on the CPU path imports,
        # is what lets the default install need only Mojo and simdjson.
        comptime assert (
            False
        ), "target='gpu' moved in 0.4.0: install max-core and call json_gpu"
    else:
        return _parse_cpu_mojo(s^)


def loads[target: StaticString = "cpu"](bytes: Span[UInt8, _]) raises -> Value:
    """Deserialize JSON bytes to a Value.

    The same parse as the string form, for callers that already hold
    bytes: a file buffer, a `List[UInt8]`, or a slice of a larger
    document. The bytes are copied once into the document that backs
    the returned `Value`, instead of being turned into a `String` by
    the caller and copied again on the way in.

    The bytes are checked for well-formed UTF-8 before anything else.
    They have to be: `String` carries that as an invariant, so handing
    it ill-formed bytes is undefined behaviour and trips an assertion
    under `-D ASSERT=all`. An ill-formed input raises here instead,
    which is the same outcome the parser would have reached for a
    string, reported earlier and by position.

    Parameters:
        target: Parsing target/backend. Options: "cpu" (default, pure Mojo),
            "cpu-simdjson" (FFI), or "gpu" (for large files).

    Args:
        bytes: JSON text to parse.

    Returns:
        Parsed Value.

    Example:
        var data = loads(buffer.as_bytes()).
    """
    if not is_valid_utf8(bytes):
        raise Error("invalid UTF-8 in input")
    return _loads_value[target](String(unsafe_from_utf8=bytes))


def loads[
    target: StaticString = "cpu"
](var s: String, config: ParserConfig) raises -> Value:
    """Deserialize JSON with custom configuration.

    Parameters:
        target: "cpu" (default), "cpu-simdjson", or "gpu".

    Args:
        s: JSON string to parse.
        config: Parser configuration (allow_comments, allow_trailing_comma, max_depth).

    Returns:
        Parsed Value.

    Example:
        var data = loads('{"a": 1} // comment', ParserConfig(allow_comments=True)).
    """

    # The surrogate rule reads the source, because parsing turns an
    # unpaired escape into U+FFFD and that is indistinguishable from a
    # literal one. The uniqueness rule reads the parse, because it is
    # about structure. Both run only when asked for.
    if config.ijson:
        check_no_unpaired_surrogates(s)
    var preprocessed = preprocess_json(s, config)
    var value = _loads_value[target](preprocessed^)
    if config.ijson:
        check_unique_member_names(value)
    return value^


def loads[
    target: StaticString = "cpu",
    format: StaticString = "json",
](var s: String) raises -> List[Value]:
    """Deserialize NDJSON string to a list of Values.

    Parameters:
        target: "cpu" (default), "cpu-simdjson", or "gpu".
        format: Must be "ndjson" for this overload.

    Args:
        s: NDJSON string (one JSON value per line).

    Returns:
        List of parsed Values.

    Example:
        var values = loads[format="ndjson"]('{"a":1}\\n{"a":2}').
    """

    comptime if format != "ndjson":
        comptime assert False, "Use format='ndjson' for List[Value] return type"

    var result = List[Value]()
    var lines = _split_lines(s)

    for i in range(len(lines)):
        var line = lines[i]
        if _is_whitespace_only(line):
            continue
        var value = _loads_value[target](line)
        result.append(value^)

    return result^


def loads[lazy: Bool](s: String) raises -> LazyValue:
    """Create a lazy JSON value for on-demand parsing (CPU only).

    Parameters:
        lazy: Must be True (required, no default).

    Args:
        s: JSON string.

    Returns:
        LazyValue that parses on demand.

    Example:
        var lazy = loads[lazy=True](huge_json)
        var name = lazy.get("/users/0/name")  # Only parses this path.

    Note:
        Lazy parsing is CPU-only. For GPU, use `loads[target="gpu"]` directly.
    """

    comptime if not lazy:
        comptime assert False, "Use lazy=True for LazyValue return type"

    return LazyValue(s)


def load[target: StaticString = "cpu"](mut f: FileHandle) raises -> Value:
    """Deserialize JSON from file to a Value (like Python's json.load).

    Parameters:
        target: "cpu" (default), "cpu-simdjson", or "gpu".

    Args:
        f: FileHandle to read JSON from.

    Returns:
        Parsed Value.

    Example:
        with open("data.json", "r") as f:
            var data = load(f).
    """
    var content = f.read()
    return loads[target](content)


def load[
    target: StaticString = "cpu"
](mut f: FileHandle, config: ParserConfig) raises -> Value:
    """Deserialize JSON from file with custom configuration.

    Parameters:
        target: "cpu" (default), "cpu-simdjson", or "gpu".

    Args:
        f: FileHandle to read JSON from.
        config: Parser configuration.

    Returns:
        Parsed Value.
    """
    var content = f.read()
    return loads[target](content, config)


def load[target: StaticString = "cpu"](path: String) raises -> Value:
    """Load JSON/NDJSON from file path. Auto-detects format from extension.

    Parameters:
        target: "cpu" (default), "cpu-simdjson", or "gpu".

    Args:
        path: Path to .json or .ndjson file.

    Returns:
        Value (for .json) or Value array (for .ndjson).

    Example:
        var data = load("config.json")           # Returns object/value.
        var items = load("data.ndjson")          # Returns array of values.
        var big = load[target="gpu"]("large.json").
    """
    var f = open(path, "r")
    var content = f.read()
    f.close()

    # Auto-detect NDJSON from extension
    if path.endswith(".ndjson"):
        var values = loads[target, format="ndjson"](content)
        return _list_to_array_value(values)

    return loads[target](content)


def _list_to_array_value(values: List[Value]) raises -> Value:
    """Convert `List[Value]` to a tape-backed array `Value`.

    Serializes each element to JSON and re-parses the joined `[ ... ]`
    payload through the canonical CPU pipeline so the result is a
    standard tape-backed view.
    """
    var count = len(values)
    if count == 0:
        return loads("[]")

    var raw = String("[")
    for i in range(count):
        if i > 0:
            raw += ","
        raw += dumps(values[i])
    raw += "]"
    return loads(raw)


def load[
    target: StaticString = "cpu",
    format: StaticString = "json",
](path: String) raises -> List[Value]:
    """Load NDJSON from file path -> `List[Value]` (matches `loads[format='ndjson']`).

    Parameters:
        target: "cpu" (default), "cpu-simdjson", or "gpu".
        format: Must be "ndjson" for this overload.

    Args:
        path: Path to a `.ndjson` file.

    Returns:
        List of parsed Values (one per non-empty line).

    Example:
        var events = load[format="ndjson"]("logs.ndjson")
        for event in events:
            print(event["msg"].string_value()).

    Note:
        This overload returns `List[Value]`, matching
        `loads[format='ndjson']`. The plain-extension auto-detection
        (`load("x.ndjson") -> Value`) still works for backward
        compatibility but wraps the records in an array Value; new
        code should prefer this typed overload.
    """

    comptime if format != "ndjson":
        comptime assert False, "Use format='ndjson' for List[Value] return type"

    var f = open(path, "r")
    var content = f.read()
    f.close()
    return loads[target, format="ndjson"](content)


def load[streaming: Bool](path: String) raises -> StreamingParser:
    """Stream large files line by line (CPU only, for memory efficiency).

    Parameters:
        streaming: Must be True.

    Args:
        path: Path to NDJSON file.

    Returns:
        StreamingParser iterator.

    Example:
        var parser = load[streaming=True]("huge.ndjson")
        while parser.has_next():
            var item = parser.next()
        parser.close().

    Note:
        Streaming is CPU-only (for memory efficiency, not speed).
        For GPU speed on files that fit in memory, use `load[target="gpu"]("file.ndjson")`.
    """

    comptime if not streaming:
        comptime assert False, "Use streaming=True for StreamingParser"

    return StreamingParser(path)


from .config import ParserConfig
from .ijson import check_no_unpaired_surrogates, check_unique_member_names
from .lazy import LazyValue
from .streaming import StreamingParser
from .errors import json_parse_error
from .unicode import unescape_json_string, unescape_json_string_span
from .config import preprocess_json
from .ndjson import _split_lines, _is_whitespace_only
