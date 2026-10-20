# json CPU backends.
#
# Two CPU paths, both producing tape-backed `Value` views:
#   - `parse_cpu_native_tape` (default): two-pass stage 1 + stage 2
#       walker that emits a `Document` and wraps the root as a
#       `Value` view. Stage 1 has scalar (`stage1_scalar`) and SIMD
#       (`stage1`) implementations; both are byte-identical, enforced
#       by `tests/test_stage1_equivalence.mojo`. Default is SIMD
#       (1.5x to 2.2x faster on the benchmark corpora); opt into the
#       scalar oracle with `parse_cpu_native_tape[force_scalar=True]`.
#   - simdjson FFI: optional C++ backend, exposed via `SimdjsonFFI`
#       and selected with `loads[target='cpu-simdjson']`. The FFI
#       output is translated into the same tape representation.

# Common type tags shared with the simdjson FFI bindings.
from .types import (
    JSON_TYPE_NULL,
    JSON_TYPE_BOOL,
    JSON_TYPE_INT64,
    JSON_TYPE_UINT64,
    JSON_TYPE_DOUBLE,
    JSON_TYPE_STRING,
    JSON_TYPE_ARRAY,
    JSON_TYPE_OBJECT,
    JSON_OK,
    JSON_ERROR_INVALID,
    JSON_ERROR_CAPACITY,
    JSON_ERROR_UTF8,
    JSON_ERROR_OTHER,
)

# simdjson FFI backend.
from .simdjson_ffi import (
    SimdjsonFFI,
    SimdjsonTape,
    SIMDJSON_OK,
    SIMDJSON_ERROR_INVALID_JSON,
    SIMDJSON_ERROR_CAPACITY,
    SIMDJSON_ERROR_UTF8,
    SIMDJSON_ERROR_OTHER,
    SIMDJSON_TYPE_NULL,
    SIMDJSON_TYPE_BOOL,
    SIMDJSON_TYPE_INT64,
    SIMDJSON_TYPE_UINT64,
    SIMDJSON_TYPE_DOUBLE,
    SIMDJSON_TYPE_STRING,
    SIMDJSON_TYPE_ARRAY,
    SIMDJSON_TYPE_OBJECT,
)

from std.memory import ArcPointer
from ..value import Value
from ..value.value import make_view_value
from ..document import Document
from .stage1_scalar import (
    parse_structural_scalar,
    StructuralIndex,
)
from .stage1 import parse_structural_simd
from .stage2 import parse_two_pass_tape


def parse_cpu_native_tape[
    force_scalar: Bool = False
](var s: String) raises -> Value:
    """Two-pass CPU parser that returns a tape-backed `Value` view.

    Stage 1 finds structural positions (SIMD by default; scalar with
    `force_scalar=True`); stage 2 walks the resulting index and emits
    a `Document` whose root is wrapped as a `Value` view. Reads
    (`is_*`, `*_value`, `array_count`, `__getitem__`, etc.) hit the
    tape directly; mutations materialise into a temporary owned tree
    and rebuild the document transparently.

    Parameters:
        force_scalar: SIMD vs scalar stage 1 (default SIMD).

    Args:
        s: JSON input string. The returned `Value`'s document owns
            this string.

    Returns:
        Tape-backed `Value` view of the document root.
    """
    var doc = parse_two_pass_tape[force_scalar=force_scalar](s^)
    var root_idx = doc.root()
    var arc = ArcPointer[Document](doc^)
    return make_view_value(arc, root_idx)
