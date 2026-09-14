# GPU tape adapter.
#
# Converts a `JSONResult` produced by the GPU kernel plus the original
# input bytes into a `Value` by feeding stage 2 of the CPU pipeline.
#
# Value construction is centralised in `cpu/stage2.mojo`, which walks
# a `StructuralIndex`. This adapter bridges the GPU output to that
# pipeline so CPU and GPU paths produce identical tape layouts.
#
# Reusing GPU work
# ----------------
# The GPU kernel emits structural positions for `{` `}` `[` `]` `:` `,`
# only -- quotes are tracked internally but not emitted. Stage 2 needs
# quote positions too (they delimit string spans). Rather than re-running
# the full scalar scan, the adapter walks the input once with a quote-only
# pass that:
#
#   - tracks `in_string` / `escaped` state to skip structural-looking
#     bytes inside string literals,
#   - emits both opening and closing quote offsets in order, and
#   - merges in the GPU-supplied `{}[]:,` positions at the byte offset
#     they describe.
#
# The merged result is identical to what `stage1_scalar.parse_structural`
# would produce on the same input. (The equivalence is asserted as a
# debug-build invariant: `_validate_against_scalar` is intentionally
# off-by-default; enable with `-D JSON_GPU_VALIDATE_INDEX=1` when chasing
# a regression.)
#
# This keeps the GPU work on the critical path: the bracket/comma scan
# stays on GPU; the quote scan is a small CPU pass; stage 2 then walks
# the merged index in O(structural_count).
#
# Licensing: MIT like the rest of this project, but building this file
# requires `max-core`, which is governed by the Modular Community
# License. See `json/gpu/LICENSE-GPU.md`.

from std.collections import List
from std.memory import ArcPointer

from ..value import Value
from ..value.value import make_view_value
from ..document import Document
from ..types import JSONResult
from ..cpu.stage1_scalar import StructuralIndex, parse_structural_scalar
from ..cpu.stage2 import parse_into_document


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_gpu_to_value(
    var input: String, gpu_result: JSONResult
) raises -> Value:
    """Convert a GPU `JSONResult` into a tape-backed `Value` via stage 2.

    Args:
        input: Original JSON bytes (the same bytes handed to the GPU
            kernel; the adapter does not re-decode them). The returned
            `Value`'s document owns these bytes.
        gpu_result: GPU output. Only `gpu_result.structural` is consumed
            here; `pair_pos` is computed but currently unused -- a future
            patch can pass it to stage 2 to skip the inner-bracket walk
            entirely.

    Returns:
        Tape-backed `Value` view of the parsed document root.
    """
    var index = _result_to_index(input, gpu_result)
    var doc = parse_into_document(input^, index)
    var root_idx = doc.root()
    var arc = ArcPointer[Document](doc^)
    return make_view_value(arc, root_idx)


# ---------------------------------------------------------------------------
# Index merge: GPU `{}[]:,` positions + CPU quote scan
# ---------------------------------------------------------------------------


def _result_to_index(
    input: String, gpu_result: JSONResult
) raises -> StructuralIndex:
    """Build a stage1-compatible `StructuralIndex` from GPU output.

    The walk is single-pass and skips the inside of string literals using
    only quote and backslash bookkeeping -- the GPU has already given us
    every `{}[]:,` outside strings, so the CPU pass does not need to
    classify those bytes.
    """
    var bytes = input.as_bytes()
    var n = len(bytes)
    var gpu_positions = gpu_result.structural.copy()
    var gpu_size = len(gpu_positions)

    var index = StructuralIndex(capacity=gpu_size + n // 16)

    var gpu_idx = 0
    var in_string = False
    var escaped = False
    var i = 0

    while i < n:
        var c = bytes[i]

        if escaped:
            # Skip the byte after a backslash. Drop any GPU position
            # that falls on it (e.g. an escaped `,` or `:` inside a
            # string body) so it does not leak out when we re-emerge.
            while gpu_idx < gpu_size and Int(gpu_positions[gpu_idx]) <= i:
                gpu_idx += 1
            escaped = False
            i += 1
            continue

        if in_string:
            # Inside a string literal: drop any GPU position that
            # falls on this byte. The GPU kernel emits the *raw*
            # `{}[]:,` bitmap (it doesn't know which bytes live
            # inside string literals), so the in-string string body
            # is exactly where we filter that noise out.
            while gpu_idx < gpu_size and Int(gpu_positions[gpu_idx]) <= i:
                gpu_idx += 1
            if c == UInt8(ord("\\")):
                escaped = True
                i += 1
                continue
            if c == UInt8(ord('"')):
                index.positions.append(UInt32(i))
                in_string = False
                i += 1
                continue
            i += 1
            continue

        if c == UInt8(ord('"')):
            index.positions.append(UInt32(i))
            in_string = True
            i += 1
            continue

        # Outside a string: emit the GPU position at this byte if
        # there is one. Defensive bounded drain handles any
        # mis-ordered GPU output (well-formed output never lags
        # behind the byte cursor, but we never trust GPU output
        # blindly).
        while gpu_idx < gpu_size and Int(gpu_positions[gpu_idx]) < i:
            index.positions.append(UInt32(gpu_positions[gpu_idx]))
            gpu_idx += 1

        if gpu_idx < gpu_size and Int(gpu_positions[gpu_idx]) == i:
            index.positions.append(UInt32(gpu_positions[gpu_idx]))
            gpu_idx += 1

        i += 1

    # Anything left after the byte cursor is past the input -- drop
    # it, since it cannot correspond to a real structural position.
    return index^


# ---------------------------------------------------------------------------
# Debug invariant (off by default)
# ---------------------------------------------------------------------------
#
# `_validate_against_scalar` checks that the merged GPU+quote index is
# byte-identical to the pure-scalar stage 1 output. It is wired into a
# `comptime if is_defined["JSON_GPU_VALIDATE_INDEX"]()` guard at the
# call site (parser.mojo) when debugging a regression. Leaving the helper
# here so test code or `-D JSON_GPU_VALIDATE_INDEX=1` builds can use it.


def _validate_against_scalar(input: String, merged: StructuralIndex) raises:
    """Raise if the merged index disagrees with the scalar oracle.

    Used only when `JSON_GPU_VALIDATE_INDEX` is defined at compile time.
    """
    var oracle = parse_structural_scalar(input)
    var a = merged.positions.copy()
    var b = oracle.positions.copy()

    if len(a) != len(b):
        raise Error(
            "tape_adapter: merged index size "
            + String(len(a))
            + " disagrees with scalar oracle size "
            + String(len(b))
        )

    for i in range(len(a)):
        if a[i] != b[i]:
            raise Error(
                "tape_adapter: merged index position "
                + String(i)
                + " disagrees with scalar oracle (got "
                + String(Int(a[i]))
                + ", want "
                + String(Int(b[i]))
                + ")"
            )
