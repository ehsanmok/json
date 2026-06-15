# Stage 2: walk a `StructuralIndex` and emit a tape-backed `Document`.
#
# Stage 1 -- either `stage1_scalar.parse_structural_scalar` or
# `stage1.parse_structural_simd` -- produces an ordered list of byte
# offsets into the input, one per structural character (plus quote
# boundaries). Stage 2 walks that list left-to-right and emits tape
# entries into a `Document` without ever re-scanning the byte stream
# for structure.
#
# Decoupling stage 1 from value construction means a different
# structural scanner (e.g. the GPU pipeline) can drop in without
# touching this module. Children are always written before parents
# so a parent's `child_start_idx` payload points backwards into a
# contiguous run of header entries; the root is the last entry,
# which is what `Document.root()` assumes.
#
# The walker is iterative: `parse_into_document` maintains an
# explicit `List[_Frame]` work stack instead of recursing per
# container, dispatches values inline, and only calls the primitive
# helpers (`_emit_string`, `_emit_number`, `_parse_object_key`) that
# cannot recurse. A shared `headers_scratch: List[UInt64]` collects
# each container's children's headers and is `memcpy`'d into
# `doc.tape` (and shrunk back) on container close, so we get a
# single amortized allocation over the whole document rather than
# one per container.
#
# `_find_matching_close` is avoided entirely: `positions` is
# monotonically increasing, so the next `]` / `}` for the current
# container is just the next position whose byte is `]` / `}` after
# the children have been fully consumed.
#
# Validation rules (trailing commas, double commas, leading commas,
# missing colons, missing values after colons, unquoted keys) all
# raise structured errors during the walk; coverage in
# `tests/test_stage2_tape.mojo`.

from std.bit import count_trailing_zeros
from std.collections import List
from std.memory import memcpy
from std.memory.unsafe import pack_bits
from std.sys import simd_byte_width


# Native SIMD chunk size in bytes for stage 2's helpers (`_skip_ws`,
# `_string_has_escape`). Picking the host's vector register width
# means a single load + eq + reduce per iteration on every backend:
#
#   * NEON (Apple Silicon, 128-bit): 16 bytes
#   * AVX2 (most x86):              32 bytes
#   * AVX-512 (Sapphire Rapids+):   64 bytes
#
# Mojo's SIMD docs explicitly recommend not exceeding 2x register
# width or "the resulting code will perform poorly", so a hard-coded
# 32 was a 2x penalty on NEON. simd_byte_width() defers to the
# Mojo compiler's per-target answer.
comptime _BLOCK: Int = simd_byte_width()

from ..unicode import unescape_json_string_span
from ..document import (
    Document,
    pack_tape_entry,
    pack_pair,
    TAPE_TAG_NULL,
    TAPE_TAG_BOOL,
    TAPE_TAG_INT,
    TAPE_TAG_FLOAT,
    TAPE_TAG_STRING,
    TAPE_TAG_STRING_OWNED,
    TAPE_TAG_ARRAY,
    TAPE_TAG_OBJECT,
    TAPE_TAG_KEY,
    TAPE_TAG_KEY_INLINE,
)
from .stage1_scalar import StructuralIndex
from .number_parse import parse_int_swar


# ---------------------------------------------------------------------------
# Whitespace + primitive end helpers
# ---------------------------------------------------------------------------


@always_inline
def _string_has_escape(bytes: Span[UInt8, _], start: Int, end: Int) -> Bool:
    """SIMD-accelerated scan for backslash in [start, end).

    The dominant case on real JSON corpora is "no escape", so we scan
    `_BLOCK` bytes at a time (16 on NEON, 32 on AVX2, 64 on AVX-512)
    using `SIMD.eq` + `reduce_or`. Each iteration is a single native
    SIMD load + eq + reduce; using more than the host register width
    would force the compiler to split into multiple instructions.
    Tail (< _BLOCK bytes) is handled scalar.
    """
    var n = end - start
    if n <= 0:
        return False
    var ptr = bytes.unsafe_ptr()
    var i = start
    var stop = end - _BLOCK
    while i <= stop:
        var chunk = ptr.load[width=_BLOCK](i)
        if chunk.eq(UInt8(ord("\\"))).reduce_or():
            return True
        i += _BLOCK
    while i < end:
        if ptr[i] == UInt8(ord("\\")):
            return True
        i += 1
    return False


@always_inline
def _is_ws(b: UInt8) -> Bool:
    return (
        b == UInt8(ord(" "))
        or b == UInt8(ord("\t"))
        or b == UInt8(ord("\n"))
        or b == UInt8(ord("\r"))
    )


@always_inline
def _skip_ws(bytes: Span[UInt8, _], start: Int, end: Int) -> Int:
    """Skip ASCII whitespace.

    Branch-prediction-friendly hybrid:
      * The first ~4 bytes are checked scalar (the dense-JSON case
        where there is 0-1 whitespace bytes between fields - no SIMD
        setup cost).
      * Only if we're still in whitespace after that do we fall into
        the SIMD `_BLOCK`-byte loop, which uses pack_bits +
        count_trailing_zeros to find the first non-whitespace byte
        without per-byte iteration.

    `_BLOCK` is `simd_byte_width()`, so each iteration is a single
    native SIMD instruction on every backend (16 B NEON, 32 B AVX2,
    64 B AVX-512) instead of a 2x or 4x register-width split.

    Pretty-printed corpora (citm: ~9 ws bytes/pair) hit the SIMD path;
    compact corpora pay only a single scalar byte test.
    """
    var i = start
    var ptr = bytes.unsafe_ptr()

    # Scalar prelude (up to 4 bytes). Common case for compact / dense
    # JSON: returns after the first byte test.
    var scalar_stop = min(end, start + 4)
    while i < scalar_stop and _is_ws(ptr[i]):
        i += 1
    if i < scalar_stop:
        return i

    # SIMD body for long whitespace runs.
    var stop = end - _BLOCK
    while i <= stop:
        var chunk = ptr.load[width=_BLOCK](i)
        var is_ws_mask = (
            chunk.eq(UInt8(ord(" ")))
            | chunk.eq(UInt8(ord("\t")))
            | chunk.eq(UInt8(ord("\n")))
            | chunk.eq(UInt8(ord("\r")))
        )

        # pack_bits' output dtype must hold _BLOCK bits. Comptime
        # branch picks the right width with no runtime cost.
        comptime if _BLOCK == 16:
            var ws_bits = pack_bits[dtype=DType.uint16](is_ws_mask)
            if ws_bits != UInt16(0xFFFF):
                var first_non_ws = Int(count_trailing_zeros(~ws_bits))
                return i + first_non_ws
        elif _BLOCK == 32:
            var ws_bits = pack_bits[dtype=DType.uint32](is_ws_mask)
            if ws_bits != UInt32(0xFFFF_FFFF):
                var first_non_ws = Int(count_trailing_zeros(~ws_bits))
                return i + first_non_ws
        elif _BLOCK == 64:
            var ws_bits = pack_bits[dtype=DType.uint64](is_ws_mask)
            if ws_bits != UInt64(0xFFFF_FFFF_FFFF_FFFF):
                var first_non_ws = Int(count_trailing_zeros(~ws_bits))
                return i + first_non_ws
        else:
            comptime assert False, "unsupported simd_byte_width()"
        i += _BLOCK

    while i < end and _is_ws(ptr[i]):
        i += 1
    return i


def _primitive_end(bytes: Span[UInt8, _], start: Int, end: Int) -> Int:
    """First byte after a top-level primitive (number / null / true /
    false). Skips leading whitespace then scans until whitespace, EOF,
    or a structural byte."""
    var i = _skip_ws(bytes, start, end)
    while i < end:
        var c = bytes[i]
        if (
            _is_ws(c)
            or c == UInt8(ord(","))
            or c == UInt8(ord("}"))
            or c == UInt8(ord("]"))
        ):
            break
        i += 1
    return i


def _validate_escapes(bytes: Span[UInt8, _], start: Int, end: Int) raises:
    var j = start
    while j < end:
        if bytes[j] != UInt8(ord("\\")):
            j += 1
            continue
        if j + 1 >= end:
            raise Error("Stage 2: trailing backslash in string")
        var esc = bytes[j + 1]
        if (
            esc != UInt8(ord('"'))
            and esc != UInt8(ord("\\"))
            and esc != UInt8(ord("/"))
            and esc != UInt8(ord("b"))
            and esc != UInt8(ord("f"))
            and esc != UInt8(ord("n"))
            and esc != UInt8(ord("r"))
            and esc != UInt8(ord("t"))
            and esc != UInt8(ord("u"))
        ):
            raise Error(
                "Stage 2: invalid escape sequence '\\"
                + chr(Int(esc))
                + "' at offset "
                + String(j)
            )
        if esc == UInt8(ord("u")):
            j += 6
        else:
            j += 2


# ---------------------------------------------------------------------------
# Convenience: full parse from a raw input string into a tape-backed
# `Document`.
# ---------------------------------------------------------------------------


def parse_two_pass_tape[
    force_scalar: Bool = False
](var input: String) raises -> Document:
    """End-to-end stage 1 + stage 2 parse that emits a `Document`.

    Parameters:
        force_scalar: When False (default), use the SIMD stage 1
            implementation (`stage1.parse_structural_simd`); on the
            benchmark corpora SIMD is ~1.2x faster than the scalar
            walker. When True, use the scalar oracle -- useful for
            differential testing and for inputs small enough that the
            SIMD chunk loop never runs (n < 32). Both produce identical
            output (enforced by `tests/test_stage1_equivalence.mojo`);
            this is purely a performance switch.

    Args:
        input: JSON input. The returned `Document` owns this string,
            so its bytes can back zero-copy string slices on the tape.

    Returns:
        Owned `Document` with the root at `Document.root()`.
    """
    from .stage1_scalar import parse_structural_scalar
    from .stage1 import parse_structural_simd

    comptime if force_scalar:
        var index = parse_structural_scalar(input)
        return parse_into_document(input^, index)
    else:
        var index = parse_structural_simd(input)
        return parse_into_document(input^, index)


# ---------------------------------------------------------------------------
# Iterative tape-emitting walker
#
# `parse_into_document` walks a `StructuralIndex` with an explicit
# work stack -- one `_Frame` per open container ancestor -- and emits
# entries directly into `doc.tape`. Primitive helpers (`_emit_string`,
# `_emit_number`, `_parse_object_key`) are defined below; they don't
# recurse, so they stay as functions.
#
# Each outer iteration does one of:
#   (a) emit a primitive at `cursor` and enter the attach phase
#   (b) open a container at `cursor` (push frame) and loop back to
#       parse the first child
#
# The inner "attach" loop walks back up the stack when a value sits
# at one or more matching closes (e.g. `]]]` after the last leaf),
# emitting each container header in turn.
# ---------------------------------------------------------------------------


struct _Frame(Copyable, Movable):
    """One open container on the work stack."""

    var kind: UInt8  # _FRAME_ARRAY or _FRAME_OBJECT
    var headers_lo: Int  # checkpoint into headers_scratch

    @always_inline
    def __init__(out self, kind: UInt8, headers_lo: Int):
        self.kind = kind
        self.headers_lo = headers_lo


comptime _FRAME_ARRAY: UInt8 = 0
comptime _FRAME_OBJECT: UInt8 = 1


def parse_into_document(
    var input: String, index: StructuralIndex
) raises -> Document:
    """Build a `Document` (tape + side pools) by walking the structural
    index over `input`. The returned document owns `input`.

    Args:
        input: Original JSON bytes.
        index: Output of stage 1. Borrowed immutably -- `positions` is
            read in place, no copy on entry.

    Returns:
        Owned `Document` whose root entry is the last tape slot.
    """
    var n = input.byte_length()
    ref positions = index.positions

    var doc = Document(input^)

    # Pre-size: each structural position contributes ~0.5 - 1 tape
    # entries (open/close/comma/colon are bookkeeping; quote pairs and
    # the primitives between commas are the entries). On twitter and
    # citm `len(tape) / len(positions)` is around 0.55, so reserving
    # `len(positions)` is a slight over-estimate that keeps the tape
    # vector from reallocating mid-walk. The +8 covers the root
    # primitive case (positions can be empty for `42`).
    doc.tape.reserve(len(positions) + 8)
    var side_hint = len(positions) // 8 + 4
    doc.float_pool.reserve(side_hint)
    doc.string_pool.reserve(side_hint)
    doc.key_pool.reserve(side_hint)

    # Shared scratch for collecting children's headers across all
    # frames. Each frame uses a contiguous slice `[lo, len(scratch))`
    # and shrinks back to `lo` at close.
    var headers_scratch = List[UInt64]()
    headers_scratch.reserve(64)

    var stack = List[_Frame]()
    stack.reserve(32)

    var pos_idx = 0
    var doc_start = _skip_ws(doc.input.as_bytes(), 0, n)
    var cursor = doc_start

    var root_header: UInt64 = 0
    var root_value_end = doc_start
    var have_root = False

    while not have_root:
        # --- PARSE_VALUE phase ----------------------------------------
        # Emit a primitive, OR open a container (push frame + continue).
        var bytes = doc.input.as_bytes()
        var i = _skip_ws(bytes, cursor, n)
        if i >= n:
            raise Error("Stage 2: empty value")

        var c = bytes[i]
        var value_header: UInt64
        var value_end: Int

        if c == UInt8(ord("{")):
            if pos_idx >= len(positions) or Int(positions[pos_idx]) != i:
                raise Error("Stage 2: cursor desync at object open")
            pos_idx += 1

            var headers_lo = len(headers_scratch)
            var after_open = _skip_ws(bytes, i + 1, n)
            var bytes_ref = doc.input.as_bytes()

            if (
                pos_idx < len(positions)
                and Int(positions[pos_idx]) == after_open
                and after_open < n
                and bytes_ref[after_open] == UInt8(ord("}"))
            ):
                # Empty object fast path -- no frame push.
                pos_idx += 1
                var child_start = len(doc.tape)
                value_header = pack_tape_entry(
                    TAPE_TAG_OBJECT,
                    pack_pair(UInt64(0), UInt64(child_start)),
                )
                value_end = after_open + 1
                # Fall through to ATTACH phase below.
            else:
                if after_open >= n:
                    raise Error("Stage 2: unterminated object")
                if bytes_ref[after_open] == UInt8(ord(",")):
                    raise Error(
                        "Stage 2: leading comma in object at offset "
                        + String(after_open)
                    )

                stack.append(_Frame(_FRAME_OBJECT, headers_lo))
                cursor = _parse_object_key(
                    doc, positions, pos_idx, headers_scratch, after_open, n
                )
                continue  # outer loop: parse the value at cursor

        elif c == UInt8(ord("[")):
            if pos_idx >= len(positions) or Int(positions[pos_idx]) != i:
                raise Error("Stage 2: cursor desync at array open")
            pos_idx += 1

            var headers_lo = len(headers_scratch)
            var after_open = _skip_ws(bytes, i + 1, n)
            var bytes_ref = doc.input.as_bytes()

            if (
                pos_idx < len(positions)
                and Int(positions[pos_idx]) == after_open
                and after_open < n
                and bytes_ref[after_open] == UInt8(ord("]"))
            ):
                # Empty array fast path -- no frame push.
                pos_idx += 1
                var child_start = len(doc.tape)
                value_header = pack_tape_entry(
                    TAPE_TAG_ARRAY,
                    pack_pair(UInt64(0), UInt64(child_start)),
                )
                value_end = after_open + 1
                # Fall through to ATTACH phase below.
            else:
                if after_open >= n:
                    raise Error("Stage 2: unterminated array")
                if bytes_ref[after_open] == UInt8(ord(",")):
                    raise Error(
                        "Stage 2: leading comma in array at offset "
                        + String(after_open)
                    )

                stack.append(_Frame(_FRAME_ARRAY, headers_lo))
                cursor = after_open
                continue  # outer loop: parse the value at cursor

        elif c == UInt8(ord('"')):
            var v_end = i
            value_header = _emit_string(doc, positions, pos_idx, v_end, i)
            value_end = v_end

        elif c == UInt8(ord("n")):
            if (
                i + 4 > n
                or bytes[i + 1] != UInt8(ord("u"))
                or bytes[i + 2] != UInt8(ord("l"))
                or bytes[i + 3] != UInt8(ord("l"))
            ):
                raise Error(
                    "Stage 2: expected 'null' literal at offset " + String(i)
                )
            value_header = pack_tape_entry(TAPE_TAG_NULL, 0)
            value_end = i + 4

        elif c == UInt8(ord("t")):
            if (
                i + 4 > n
                or bytes[i + 1] != UInt8(ord("r"))
                or bytes[i + 2] != UInt8(ord("u"))
                or bytes[i + 3] != UInt8(ord("e"))
            ):
                raise Error(
                    "Stage 2: expected 'true' literal at offset " + String(i)
                )
            value_header = pack_tape_entry(TAPE_TAG_BOOL, 1)
            value_end = i + 4

        elif c == UInt8(ord("f")):
            if (
                i + 5 > n
                or bytes[i + 1] != UInt8(ord("a"))
                or bytes[i + 2] != UInt8(ord("l"))
                or bytes[i + 3] != UInt8(ord("s"))
                or bytes[i + 4] != UInt8(ord("e"))
            ):
                raise Error(
                    "Stage 2: expected 'false' literal at offset " + String(i)
                )
            value_header = pack_tape_entry(TAPE_TAG_BOOL, 0)
            value_end = i + 5

        elif c == UInt8(ord("-")) or (
            c >= UInt8(ord("0")) and c <= UInt8(ord("9"))
        ):
            var v_end = i
            value_header = _emit_number(doc, v_end, i, n)
            value_end = v_end

        else:
            raise Error("Stage 2: unexpected character at offset " + String(i))

        # --- ATTACH phase ---------------------------------------------
        # Attach `value_header` to its parent (or set as root). After
        # a container close the container's own header becomes the
        # new pending value, so we may chain through several closes
        # (e.g. `]]]`) before parsing the next sibling.
        while True:
            if len(stack) == 0:
                root_header = value_header
                root_value_end = value_end
                have_root = True
                break  # inner; outer exits because have_root is set

            headers_scratch.append(value_header)

            var bytes2 = doc.input.as_bytes()
            var j = _skip_ws(bytes2, value_end, n)
            var top = stack[len(stack) - 1].copy()
            var is_array = top.kind == _FRAME_ARRAY

            if j >= n:
                if is_array:
                    raise Error("Stage 2: unterminated array")
                else:
                    raise Error("Stage 2: unterminated object")

            var b = bytes2[j]
            var matching_close = (
                UInt8(ord("]")) if is_array else UInt8(ord("}"))
            )

            if b == matching_close:
                if pos_idx >= len(positions) or Int(positions[pos_idx]) != j:
                    if is_array:
                        raise Error("Stage 2: cursor desync at array close")
                    else:
                        raise Error("Stage 2: cursor desync at object close")
                pos_idx += 1

                # Flush this frame's children (scratch[headers_lo:])
                # into the tape and shrink scratch back to the
                # checkpoint.
                var count = len(headers_scratch) - top.headers_lo
                var child_start = len(doc.tape)
                if count > 0:
                    doc.tape.resize(child_start + count, 0)
                    memcpy(
                        dest=doc.tape.unsafe_ptr() + child_start,
                        src=headers_scratch.unsafe_ptr() + top.headers_lo,
                        count=count,
                    )
                headers_scratch.shrink(top.headers_lo)

                if is_array:
                    value_header = pack_tape_entry(
                        TAPE_TAG_ARRAY,
                        pack_pair(UInt64(count), UInt64(child_start)),
                    )
                else:
                    var pair_count = count // 2
                    value_header = pack_tape_entry(
                        TAPE_TAG_OBJECT,
                        pack_pair(UInt64(pair_count), UInt64(child_start)),
                    )

                value_end = j + 1
                _ = stack.pop()
                # Continue inner loop: this container header is now
                # the pending value and attaches to its own parent.
                continue

            if b != UInt8(ord(",")):
                if is_array:
                    raise Error(
                        "Stage 2: expected ',' or ']' in array at offset "
                        + String(j)
                    )
                else:
                    raise Error(
                        "Stage 2: expected ',' or '}' in object at offset "
                        + String(j)
                    )

            if pos_idx >= len(positions) or Int(positions[pos_idx]) != j:
                if is_array:
                    raise Error("Stage 2: cursor desync at array comma")
                else:
                    raise Error("Stage 2: cursor desync at object comma")
            pos_idx += 1

            var next_cursor = _skip_ws(bytes2, j + 1, n)
            if next_cursor >= n:
                if is_array:
                    raise Error(
                        "Stage 2: trailing comma in array at offset "
                        + String(j)
                    )
                else:
                    raise Error(
                        "Stage 2: trailing comma in object at offset "
                        + String(j)
                    )
            var nb = bytes2[next_cursor]
            if nb == matching_close:
                if is_array:
                    raise Error(
                        "Stage 2: trailing comma in array at offset "
                        + String(j)
                    )
                else:
                    raise Error(
                        "Stage 2: trailing comma in object at offset "
                        + String(j)
                    )
            if is_array and nb == UInt8(ord(",")):
                raise Error(
                    "Stage 2: empty element between commas in array at"
                    " offset "
                    + String(next_cursor)
                )

            if is_array:
                cursor = next_cursor
            else:
                cursor = _parse_object_key(
                    doc, positions, pos_idx, headers_scratch, next_cursor, n
                )
            break  # inner; outer parses next value at cursor

    # Trailing-content check: nothing but whitespace allowed after the
    # top-level value.
    var bytes_final = doc.input.as_bytes()
    var t = root_value_end
    while t < n:
        if not _is_ws(bytes_final[t]):
            raise Error(
                "Stage 2: trailing content after top-level JSON value at"
                " offset "
                + String(t)
            )
        t += 1

    # Root is the last appended entry.
    doc.tape.append(root_header)
    return doc^


# ---------------------------------------------------------------------------
# Non-recursive emitters used by `parse_into_document`.
# Each helper:
#   - reads `positions[pos_idx..]` and `bytes` starting at `start`,
#   - returns this value's own header (the caller decides whether to
#     append it to `doc.tape` directly or stash it in
#     `headers_scratch`),
#   - advances `pos_idx` and `value_end` so the caller knows how far
#     into the input it consumed.
# ---------------------------------------------------------------------------


def _parse_object_key(
    mut doc: Document,
    positions: List[UInt32],
    mut pos_idx: Int,
    mut headers_scratch: List[UInt64],
    cursor: Int,
    n: Int,
) raises -> Int:
    """Parse one OBJECT key + colon starting at `cursor` (must point at
    an opening quote). Appends the key header (TAPE_TAG_KEY or
    TAPE_TAG_KEY_INLINE) to `headers_scratch`, advances `pos_idx`
    past both key-quote positions plus the colon position, and
    returns the byte offset where the value starts (whitespace
    already skipped). Error messages match the recursive emitter so
    `tests/test_stage2_tape.mojo` stays green."""
    var bytes = doc.input.as_bytes()
    if bytes[cursor] != UInt8(ord('"')):
        raise Error(
            "Stage 2: expected string key at offset " + String(cursor)
        )

    if pos_idx + 1 >= len(positions) or Int(positions[pos_idx]) != cursor:
        raise Error("Stage 2: cursor desync at object key")
    var key_close = Int(positions[pos_idx + 1])
    pos_idx += 2

    var key_start = cursor + 1
    var key_len = key_close - key_start
    var has_escape = _string_has_escape(bytes, key_start, key_close)

    var key_header: UInt64
    if has_escape:
        # Slow path: keys with escapes still need allocation +
        # interning. They are rare on real JSON corpora.
        _validate_escapes(bytes, key_start, key_close)
        var unesc = unescape_json_string_span(bytes, key_start, key_close)
        var key = String(unsafe_from_utf8=unesc^)
        var key_pool_idx = len(doc.key_pool)
        doc.key_pool.append(key^)
        key_header = pack_tape_entry(TAPE_TAG_KEY, UInt64(key_pool_idx))
    else:
        # Fast path: zero-copy KEY_INLINE -- store (offset, length)
        # into input. No allocation, no memcpy, no key_pool append.
        key_header = pack_tape_entry(
            TAPE_TAG_KEY_INLINE,
            pack_pair(UInt64(key_start), UInt64(key_len)),
        )

    var after_key = _skip_ws(bytes, key_close + 1, n)
    if after_key >= n or bytes[after_key] != UInt8(ord(":")):
        raise Error(
            "Stage 2: missing ':' between key and value at offset "
            + String(after_key)
        )
    var value_start = _skip_ws(bytes, after_key + 1, n)
    if value_start >= n:
        raise Error(
            "Stage 2: missing value after ':' at offset " + String(after_key)
        )

    headers_scratch.append(key_header)

    if pos_idx >= len(positions) or Int(positions[pos_idx]) != after_key:
        raise Error("Stage 2: cursor desync at object colon")
    pos_idx += 1

    return value_start


def _emit_string(
    mut doc: Document,
    positions: List[UInt32],
    mut pos_idx: Int,
    mut value_end: Int,
    open_quote: Int,
) raises -> UInt64:
    """Emit a STRING (clean, zero-copy slice into input) or
    STRING_OWNED (post-unescape copy in `string_pool`) header.
    Reads `positions` in place; advances `pos_idx` past the open and
    close quotes."""
    if pos_idx >= len(positions) or Int(positions[pos_idx]) != open_quote:
        raise Error(
            "Stage 2: cursor desync at string open offset " + String(open_quote)
        )
    pos_idx += 1
    if pos_idx >= len(positions):
        raise Error(
            "Stage 2: unterminated string at offset " + String(open_quote)
        )
    var close_quote = Int(positions[pos_idx])
    pos_idx += 1
    value_end = close_quote + 1

    var start_idx = open_quote + 1
    var end_idx = close_quote

    var bytes = doc.input.as_bytes()
    var has_escape = _string_has_escape(bytes, start_idx, end_idx)

    if not has_escape:
        return pack_tape_entry(
            TAPE_TAG_STRING,
            pack_pair(UInt64(start_idx), UInt64(end_idx - start_idx)),
        )

    _validate_escapes(bytes, start_idx, end_idx)

    var unescaped = unescape_json_string_span(bytes, start_idx, end_idx)
    var s = String(unsafe_from_utf8=unescaped^)
    var pool_idx = len(doc.string_pool)
    doc.string_pool.append(s^)
    return pack_tape_entry(TAPE_TAG_STRING_OWNED, UInt64(pool_idx))


def _emit_number(
    mut doc: Document,
    mut value_end: Int,
    start: Int,
    n: Int,
) raises -> UInt64:
    """Parse a JSON number starting at `start`. Inlines small ints in
    the 60-bit tape payload; large ints and floats spill to side
    pools."""
    var bytes = doc.input.as_bytes()
    var i = start
    var is_float = False
    if bytes[i] == UInt8(ord("-")):
        i += 1

    if (
        i < n
        and bytes[i] == UInt8(ord("0"))
        and i + 1 < n
        and bytes[i + 1] >= UInt8(ord("0"))
        and bytes[i + 1] <= UInt8(ord("9"))
    ):
        raise Error(
            "Stage 2: leading zeros are not allowed in JSON numbers (offset "
            + String(start)
            + ")"
        )

    while i < n:
        var c = bytes[i]
        if c >= UInt8(ord("0")) and c <= UInt8(ord("9")):
            i += 1
            continue
        if c == UInt8(ord(".")) or c == UInt8(ord("e")) or c == UInt8(ord("E")):
            is_float = True
            i += 1
            continue
        if c == UInt8(ord("+")) or c == UInt8(ord("-")):
            i += 1
            continue
        break

    value_end = i

    if is_float:
        var num_str = String(unsafe_from_utf8=bytes[start:i])
        var pool_idx = len(doc.float_pool)
        doc.float_pool.append(atof(num_str))
        return pack_tape_entry(TAPE_TAG_FLOAT, UInt64(pool_idx))
    var v = parse_int_swar(bytes, start, i)
    var payload = UInt64(v) & ((UInt64(1) << 60) - 1)
    return pack_tape_entry(TAPE_TAG_INT, payload)
