# GPU entry points.
#
# Licensing: this file is MIT like the rest of the library, but using
# it requires `max-core`, which is governed by the Modular Community
# License. See `json/gpu/LICENSE-GPU.md`.
#
# `loads` and `load` here take the same `target` parameter as the ones
# in `json`, so switching backends is a change of import rather than a
# change of call. They live here rather than in `json/parser.mojo` for
# a mechanical
# reason. Mojo resolves every import statement it can see, whether or
# not the branch containing it survives `comptime if`, and a
# module-scope `comptime if` is rejected outright ("'comptime if' must
# be contained in a function"). So there is no way to write a
# conditional import: any mention of `.gpu` inside `parser.mojo` makes
# `max-core` a hard requirement of `import json`. An unimported
# submodule, on the other hand, is never compiled. Putting the entry
# point in a module the CPU path does not import is therefore the only
# construction that keeps the default install free of MAX.

from std.collections import List
from std.memory import unsafe_memcpy

from ..errors import json_parse_error
from ..parser import (
    _is_whitespace_only,
    _list_to_array_value,
    _split_lines,
    load as cpu_load,
    loads as cpu_loads,
    parse_number_scalar,
    parse_string_scalar,
)
from ..types import JSONInput
from ..value import Value, Null
from .parser import parse_json_gpu
from .tape_adapter import parse_gpu_to_value


def loads[target: StaticString = "gpu"](var s: String) raises -> Value:
    """Deserialize a JSON string, on the GPU by default.

    Same name and same parameter as `json.loads`, so a caller switches
    backends by changing which module they import it from rather than
    by changing any call:

        from json.gpu import loads

        var data = loads[target="gpu"](huge_json)
        var small = loads[target="cpu"](little_json)

    It lives here rather than in `json/parser.mojo` because that module
    cannot mention this one. Mojo resolves an import statement wherever
    it appears, including inside a `comptime if` branch that is false,
    and a module-scope `comptime if` is rejected outright, so there is
    no conditional import to write. A single reference from the CPU
    parser would make `max-core` a hard requirement of `import json`
    for everyone; an unimported submodule is never compiled.

    Targets other than "gpu" are forwarded to the CPU parser unchanged.

    Parameters:
        target: "gpu" (default), "cpu", or "cpu-simdjson".

    Args:
        s: JSON string to parse.

    Returns:
        Parsed Value, indistinguishable from the CPU result.

    Raises:
        Error: On malformed input, or if no accelerator is available.
    """
    comptime if target == "gpu":
        return _parse_on_gpu(s^)
    else:
        return cpu_loads[target](s^)


def load[target: StaticString = "gpu"](path: String) raises -> Value:
    """Load JSON or NDJSON from a file, on the GPU by default.

    Format is taken from the extension, as in `json.load`: a `.ndjson`
    file is read one value per line and returned as an array.

    Parameters:
        target: "gpu" (default), "cpu", or "cpu-simdjson".

    Args:
        path: Path to a `.json` or `.ndjson` file.

    Returns:
        Parsed Value, or an array of values for `.ndjson`.
    """
    comptime if target != "gpu":
        return cpu_load[target](path)

    var f = open(path, "r")
    var content = f.read()
    f.close()

    if path.endswith(".ndjson"):
        var values = List[Value]()
        var lines = _split_lines(content)
        for i in range(len(lines)):
            if _is_whitespace_only(lines[i]):
                continue
            values.append(_parse_on_gpu(lines[i]))
        return _list_to_array_value(values)

    return _parse_on_gpu(content^)


def load[target: StaticString = "gpu"](mut f: FileHandle) raises -> Value:
    """`load` reading from an already-open file.

    Parameters:
        target: "gpu" (default), "cpu", or "cpu-simdjson".

    Args:
        f: FileHandle positioned at the start of one JSON value.

    Returns:
        Parsed Value.
    """
    comptime if target != "gpu":
        return cpu_load[target](f)
    return _parse_on_gpu(f.read())


def _parse_on_gpu(var s: String) raises -> Value:
    """Run the GPU pipeline over a whole document.

    The GPU computes structural positions in parallel; the tape adapter
    applies the in-string filter on the CPU side and feeds the result
    to stage 2, so `Value` construction goes through the same code path
    as the CPU backends.

    Worth it only for large documents. Kernel launch and host-to-device
    transfer dominate below roughly a hundred megabytes on a discrete
    card; see `docs/performance.md`.
    """
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
        raise Error(json_parse_error("empty input", s, 0))

    var first_char = data[start]

    # Top-level primitives short-circuit GPU launch overhead.
    if first_char == UInt8(ord("n")):
        return Value(Null())
    if first_char == UInt8(ord("t")):
        return Value(True)
    if first_char == UInt8(ord("f")):
        return Value(False)
    if first_char == 0x22:  # '"'
        return parse_string_scalar(s, start)
    if first_char == UInt8(ord("-")) or (
        first_char >= UInt8(ord("0")) and first_char <= UInt8(ord("9"))
    ):
        return parse_number_scalar(s, start)

    # Objects and arrays: GPU produces structural positions, tape adapter
    # converts them into a Value via stage 2.
    var n = len(data)
    var bytes = List[UInt8](capacity=n)
    bytes.resize(n, 0)
    unsafe_memcpy(dest=bytes.unsafe_ptr(), src=data.unsafe_ptr(), count=n)

    var input_obj = JSONInput(bytes^)
    var result = parse_json_gpu(input_obj^)

    return parse_gpu_to_value(s, result^)
