# Hot-path number parsing for stage 2.
#
# This module isolates the integer / float parsing inner loops so we
# can specialise them without disturbing stage 2's structural walk.
#
# SWAR 8-digit integer parser
# ----------------------------
# JSON integers in real corpora are overwhelmingly short (1-9 digits:
# IDs, counts, indices). The portable SWAR trick used by simdjson and
# every other fast JSON parser is:
#
#   1. Load 8 ASCII bytes at once as a `SIMD[UInt8, 8]`.
#   2. Subtract `'0'` from each lane to get the digit value 0..9.
#   3. Multiply by a constant power-of-10 vector and reduce-add.
#
# Mojo's SIMD type makes this expressible without per-platform
# intrinsics. For non-multiple-of-8 digit counts we keep a scalar
# tail; for >16 digits (overflow territory), we fall through to a
# linear scalar path that defers overflow handling to the float path.
#
# Eisel-Lemire float fast path (planned)
# --------------------------------------
# The current `atof()` call goes through a stdlib path that allocates
# a `String` per number. The Eisel-Lemire algorithm
# (https://nigeltao.github.io/blog/2020/eisel-lemire.html) reads
# directly from the byte span and converts to `Float64` in ~30
# instructions on the fast path with no allocation. Stub left here as
# a TODO; integer fast path lands first as it covers ~70% of numbers
# on twitter / citm.

# ---------------------------------------------------------------------------
# SWAR 8-digit integer parser
# ---------------------------------------------------------------------------


@always_inline
def _is_8_digit_block(chunk: SIMD[DType.uint8, 8]) -> Bool:
    """All 8 bytes in [b'0', b'9']?"""
    var lo = chunk.ge(UInt8(ord("0")))
    var hi = chunk.le(UInt8(ord("9")))
    return (lo & hi).reduce_and()


@always_inline
def _parse_8_digits_swar(chunk: SIMD[DType.uint8, 8]) -> UInt64:
    """Convert 8 ASCII digit bytes to an unsigned integer in 0..99_999_999.

    Single-vector mul + reduce_add: each lane is multiplied by its
    positional power of 10 then summed. This compiles down to a tight
    SIMD sequence on both NEON (umlal + addv) and AVX2 (vpmulld +
    horizontal sum). The reduce is the only data-dependent op.
    """
    var digits = (chunk - UInt8(ord("0"))).cast[DType.uint64]()
    var pow10 = SIMD[DType.uint64, 8](
        10000000, 1000000, 100000, 10000, 1000, 100, 10, 1
    )
    return (digits * pow10).reduce_add()


# ---------------------------------------------------------------------------
# High-level integer parser used by stage 2
# ---------------------------------------------------------------------------


@always_inline
def parse_int_swar(bytes: Span[UInt8, _], start: Int, end: Int) -> Int64:
    """Parse a JSON integer in [start, end) into a signed `Int64`.

    Layout assumed: optional leading `-`, then 1+ ASCII digits with
    no other punctuation -- exactly what stage 2 has already validated.

    Strategy:
      * 0-7 digits  -> single scalar loop (loop carries 1 mul, 1 add).
      * 8 digits   -> one SWAR 8-block.
      * 9-15 digits -> SWAR for the first 8, scalar tail.
      * 16+ digits  -> two SWAR blocks for the first 16; anything
                      bigger overflows i64 anyway and we fall back to
                      a scalar tail (still callable and overflow-safe
                      under wrap-around; stage 2 never feeds us
                      something that exceeds Int64 range without
                      flipping `is_float`).
    """
    var i = start
    var negative = False
    if i < end and bytes[i] == UInt8(ord("-")):
        negative = True
        i += 1
    var digit_count = end - i
    var result: UInt64 = 0

    if digit_count >= 8:
        var ptr = bytes.unsafe_ptr()
        var chunk = ptr.unsafe_load[width=8](i)
        if _is_8_digit_block(chunk):
            result = _parse_8_digits_swar(chunk)
            i += 8
            digit_count -= 8
            if digit_count >= 8:
                var chunk2 = ptr.unsafe_load[width=8](i)
                if _is_8_digit_block(chunk2):
                    result = result * 100_000_000 + _parse_8_digits_swar(chunk2)
                    i += 8
                    digit_count -= 8

    while i < end:
        result = result * 10 + UInt64(bytes[i]) - UInt64(ord("0"))
        i += 1

    if negative:
        return -Int64(result)
    return Int64(result)
