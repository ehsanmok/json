# Scalar stage 1: byte-by-byte structural-index builder.
#
# This is the truth oracle for stage 1. It walks the input one byte
# at a time, tracking string-vs-not-string state and escape state,
# and emits the offset of every structural character outside strings:
#
#     {  }  [  ]  :  ,  "
#
# Both opening AND closing quotes are emitted. Stage 2 uses adjacent quote
# pairs to recover string spans without re-scanning for escapes.
#
# This file is intentionally simple: stage 1's SIMD version
# (`stage1.mojo`) is validated against this oracle in
# `tests/test_stage1_equivalence.mojo`. The scalar version stays canonical
# even when SIMD ships -- a perf bug in the SIMD path can never silently
# corrupt parse output because the SIMD output is byte-checked here.

from std.collections import List


# ---------------------------------------------------------------------------
# StructuralIndex
# ---------------------------------------------------------------------------


struct StructuralIndex(Copyable):
    """Sorted list of structural-character offsets in the input.

    `positions[i]` is the byte offset of the i-th structural character.
    The list is strictly increasing.

    Stage 2 traverses the positions in order; each position points at a
    char in `{ } [ ] : , "` outside of any string literal (with both
    opening and closing quotes recorded).
    """

    var positions: List[UInt32]

    def __init__(out self, capacity: Int = 0):
        self.positions = List[UInt32](capacity=capacity)

    def copy(self) -> Self:
        var out = Self()
        out.positions = self.positions.copy()
        return out^

    def size(self) -> Int:
        return len(self.positions)


# ---------------------------------------------------------------------------
# Scalar parse
# ---------------------------------------------------------------------------


def parse_structural_scalar(input: String) -> StructuralIndex:
    """Build a structural index by walking the input byte-by-byte.

    Always succeeds: even malformed JSON produces *some* index, since
    stage 1 only tracks string state. Errors surface in stage 2 when
    the index is walked.

    Args:
        input: Raw JSON bytes.

    Returns:
        StructuralIndex with the offsets of all structural characters
        (`{` `}` `[` `]` `:` `,` `"`) that are not inside a string.
    """
    var bytes = input.as_bytes()
    var n = len(bytes)
    # Heuristic: roughly one structural per ~4 bytes for typical JSON.
    var index = StructuralIndex(capacity=n // 4)

    var in_string = False
    var escaped = False
    var i = 0

    while i < n:
        var c = bytes[i]

        if escaped:
            # The previous byte was a backslash inside a string; this byte
            # is consumed as the escape payload (\", \\, \n, \uXXXX, ...).
            escaped = False
            i += 1
            continue

        if in_string:
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

        # Outside a string.
        if c == UInt8(ord('"')):
            index.positions.append(UInt32(i))
            in_string = True
            i += 1
            continue
        if (
            c == UInt8(ord("{"))
            or c == UInt8(ord("}"))
            or c == UInt8(ord("["))
            or c == UInt8(ord("]"))
            or c == UInt8(ord(":"))
            or c == UInt8(ord(","))
        ):
            index.positions.append(UInt32(i))
        i += 1

    return index^
