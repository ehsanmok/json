# SIMD stage 1: structural-index builder, branchless 64-byte path.
#
# This is the SIMD counterpart of `stage1_scalar.parse_structural_scalar`.
# Each 64-byte chunk is processed with branchless 64-bit bit-twiddling
# instead of a per-marker scalar escape / in-string state machine:
#
#   1. Two PSHUFB-style nibble lookups classify every byte into one of
#      `\" \\ { } [ ] : ,` or "non-marker".
#   2. `pack_bits` turns the per-byte category bools into three uint64
#      bitmaps (struct, quote, bslash) for the chunk.
#   3. `find_escape_mask64` (simdjson-style branchless escape scanner)
#      computes which bytes are escaped from the bslash mask alone.
#   4. `prefix_xor64` over the unescaped-quote mask gives a "is byte i
#      inside a string?" mask for the entire chunk in one shot.
#   5. Final per-chunk emit mask = (struct outside strings) | (every
#      string-boundary quote). That mask is iterated bit-by-bit with
#      `count_trailing_zeros` -- no per-marker branches, no per-marker
#      classifier reads, no scalar escape resolver.
#
# The only data-dependent branch in the chunk loop is the
# bit-iteration loop itself, which is proportional to the total
# number of structural characters in the input -- not the number of
# bytes scanned.
#
# Cross-chunk state is two scalars (`prev_in_string` carry and
# `prev_escape` carry); both are updated by simple shifts/AND at the
# end of each iteration.
#
# Output stays byte-for-byte identical to
# `stage1_scalar.parse_structural_scalar`; the equivalence is enforced
# by `tests/test_stage1_equivalence.mojo` on every benchmark corpus.

from std.collections import List
from std.bit import count_trailing_zeros
from std.memory.unsafe import pack_bits

from .simd_clmul import prefix_xor64, find_escape_mask64
from .stage1_scalar import StructuralIndex


# ---------------------------------------------------------------------------
# SIMD parameters
# ---------------------------------------------------------------------------


comptime SIMD_WIDTH: Int = 64
"""Bytes processed per SIMD iteration. 64 lets us pack each per-chunk
classifier output into a single uint64, which is exactly the width
both the prefix-XOR and the simdjson escape-scanner work on natively."""


# Category-bit masks (matches the PSHUFB table encoding below).
comptime _CAT_QUOTE: UInt8 = 0x01
"""Bit 0 of the classifier output: byte is `\"`."""
comptime _CAT_BSLASH: UInt8 = 0x10
"""Bit 4 of the classifier output: byte is `\\`."""
comptime _CAT_STRUCT_MASK: UInt8 = 0xEE
"""Bits 1, 2, 3, 5, 6, 7 of the classifier output: byte is one of
`: [ ] , { }`."""


# ---------------------------------------------------------------------------
# PSHUFB-style nibble-lookup classifier.
#
# Given a 64-byte chunk, returns a 64-byte SIMD vector where each lane
# is the classifier byte for the corresponding input byte (0 for
# non-markers; one of the bits above for markers).
# ---------------------------------------------------------------------------


@always_inline
def _classify_chunk[
    W: SIMDLength
](chunk: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    """Two-PSHUFB nibble-lookup marker classifier."""
    comptime _LOW_TABLE = SIMD[DType.uint8, 16](
        # idx:  0     1     2     3     4     5     6     7
        0x00,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        #       8     9     A     B     C     D     E     F
        0x00,
        0x00,
        0x02,
        0x44,
        0x30,
        0x88,
        0x00,
        0x00,
    )
    comptime _HIGH_TABLE = SIMD[DType.uint8, 16](
        # idx:  0     1     2     3     4     5     6     7
        0x00,
        0x00,
        0x21,
        0x02,
        0x00,
        0x1C,
        0x00,
        0xC0,
        #       8     9     A     B     C     D     E     F
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
    )

    var low_nibble = chunk & 0x0F
    var high_nibble = chunk >> 4
    var lo_lookup = _LOW_TABLE._dynamic_shuffle(low_nibble)
    var hi_lookup = _HIGH_TABLE._dynamic_shuffle(high_nibble)
    return lo_lookup & hi_lookup


# ---------------------------------------------------------------------------
# parse_structural_simd
# ---------------------------------------------------------------------------


def parse_structural_simd(input: String) -> StructuralIndex:
    """Build a structural index using a branchless SIMD scan.

    Output is byte-for-byte identical to
    `stage1_scalar.parse_structural_scalar(input)` -- the equivalence is
    enforced by `tests/test_stage1_equivalence.mojo`, including a
    full-document run against the benchmark corpora.

    The expensive per-marker questions ("is this `\\` an escape,
    is this `\"` already escaped, is this `,` inside a string?")
    become branchless bit-twiddling on the per-chunk uint64 masks,
    leaving just one loop -- iterating set bits in the final emit
    mask with `count_trailing_zeros`.
    """
    var bytes = input.as_bytes()
    var n = len(bytes)
    var index = StructuralIndex(capacity=n // 4)

    # Carries between 64-byte chunks.
    #
    # `prev_in_string` is full-width: 0 means "we entered this chunk
    # outside any string", `~UInt64(0)` means "inside". It's used as
    # an XOR mask against `prefix_xor64` of the unescaped-quote bitmap.
    #
    # `prev_escape` is 0 or 1; bit 0 is set iff byte 0 of the next
    # chunk is escaped by a backslash that ended the prior chunk.
    var prev_in_string: UInt64 = 0
    var prev_escape: UInt64 = 0
    var i = 0

    while i + SIMD_WIDTH <= n:
        var chunk = bytes.unsafe_ptr().load[width=SIMD_WIDTH](i)

        # --- Classify ----------------------------------------------------
        comptime W = SIMD[DType.uint8, SIMD_WIDTH]
        var classified = _classify_chunk(chunk)

        var struct_bool = (classified & W(_CAT_STRUCT_MASK)).ne(W(0))
        var quote_bool = (classified & W(_CAT_QUOTE)).ne(W(0))
        var bslash_bool = (classified & W(_CAT_BSLASH)).ne(W(0))

        var struct_mask = pack_bits[dtype=DType.uint64](struct_bool)
        var quote_mask = pack_bits[dtype=DType.uint64](quote_bool)
        var bslash_mask = pack_bits[dtype=DType.uint64](bslash_bool)

        # Fast path A: chunk has no markers AT ALL and we're outside a
        # string. This is the common shape for chunks that fall in the
        # middle of long string values (URLs, base64 blobs, prose).
        # The chunk contributes nothing and the carries don't change.
        if (
            struct_mask == 0
            and quote_mask == 0
            and bslash_mask == 0
            and prev_in_string == 0
            and prev_escape == 0
        ):
            i += SIMD_WIDTH
            continue

        # Fast path B: no quotes / no backslashes / no string state.
        # Then in-string and escape masks are both zero, so emit_mask
        # collapses to `struct_mask`. Skip the prefix-XOR / escape
        # compute entirely. Common for chunks of `[1, 2, 3, ...]` /
        # `{a:1, b:2, ...}` style payloads.
        if (
            quote_mask == 0
            and bslash_mask == 0
            and prev_in_string == 0
            and prev_escape == 0
        ):
            var local_b = struct_mask
            while local_b != 0:
                var bit = Int(count_trailing_zeros(local_b))
                index.positions.append(UInt32(i + bit))
                local_b &= local_b - 1
            i += SIMD_WIDTH
            continue

        # --- Branchless escape / in-string ------------------------------
        var escape_mask = find_escape_mask64(bslash_mask, prev_escape)
        var unescaped_quotes = quote_mask & ~escape_mask

        # `prefix_xor64(q)` flips state at every quote position and
        # all bytes after, so XORing with `prev_in_string` (extended
        # to all-1s if we entered the chunk inside a string) yields a
        # mask where bit i is 1 iff byte i is inside a string at
        # position i (the opening quote itself reads as "inside",
        # which is fine because quotes aren't structurals).
        var in_string_mask = prefix_xor64(unescaped_quotes) ^ prev_in_string

        # --- Final emit mask --------------------------------------------
        # Structural characters that are outside every string, plus
        # every unescaped quote (those are the string boundaries that
        # stage 2 needs in the index). Backslashes never enter the
        # index regardless of context.
        var emit_mask = (struct_mask & ~in_string_mask) | unescaped_quotes

        # --- Carry update for next chunk --------------------------------
        # If bit 63 of in_string_mask is set we exited this chunk
        # still inside a string; carry the all-1s flag forward.
        prev_in_string = ~UInt64(0) if (in_string_mask >> UInt64(63)) & UInt64(
            1
        ) else UInt64(0)
        # `prev_escape` was already updated by `find_escape_mask64`.

        # --- Walk emit_mask ---------------------------------------------
        var local = emit_mask
        while local != 0:
            var bit = Int(count_trailing_zeros(local))
            index.positions.append(UInt32(i + bit))
            local &= local - 1

        i += SIMD_WIDTH

    # Tail: bytes that didn't fit in the last 64-byte chunk. Reuse the
    # canonical scalar walker logic; we feed it the carries from the
    # last SIMD chunk so it agrees with `parse_structural_scalar`.
    var in_string = prev_in_string != 0
    var escaped = prev_escape != 0
    while i < n:
        var c = bytes[i]
        if escaped:
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
