# GPU entry points.
#
# Licensing: this file is MIT like the rest of the library, but using
# it requires `max-core`, which is governed by the Modular Community
# License. See `json_gpu/LICENSE-GPU.md`.
#
# These live here rather than in `json/parser.mojo` for a mechanical
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

from json.errors import json_parse_error
from json.parser import (
    ParseBackend,
    _is_whitespace_only,
    _list_to_array_value,
    _split_lines,
    parse_number_scalar,
    parse_string_scalar,
)
from json.types import JSONInput
from json.value import Value, Null
from .parser import parse_json_gpu
from .tape_adapter import parse_gpu_to_value


struct Gpu(ParseBackend):
    """The GPU backend, selected with `loads[Gpu](...)`.

    A type rather than a target name because a name would have to be
    resolved inside `json/parser.mojo`, and that module cannot mention
    this one: Mojo resolves an import statement wherever it appears,
    even in a `comptime if` branch that is false, so a single reference
    would make `max-core` a requirement of `import json` for everyone.
    Naming the backend by type puts the import in the caller's hands.

    Using it requires `max-core`, which is governed by the Modular
    Community License. See `json_gpu/LICENSE-GPU.md`.

    Example:
        from json import loads, load
        from json.gpu import Gpu

        var data = loads[Gpu](huge_json)
        var file = load[Gpu]("huge.json").
    """

    @staticmethod
    def parse(var s: String) raises -> Value:
        """Parse a whole JSON document on the GPU."""
        return _parse_on_gpu(s)


def _parse_on_gpu(s: String) raises -> Value:
    """Parse JSON on the GPU, returning a tape-backed `Value`.

    The GPU computes structural positions in parallel; the tape adapter
    applies the in-string filter on the CPU side and feeds the result
    to stage 2, so `Value` construction goes through the same code path
    as the CPU backends and the result is indistinguishable from
    `loads(s)`.

    Worth it only for large documents. Kernel launch and host-to-device
    transfer dominate below roughly a hundred megabytes on a discrete
    card; see `docs/performance.md`.

    Args:
        s: JSON text to parse.

    Returns:
        The parsed value.

    Raises:
        Error: On malformed input, or if no accelerator is available.
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
