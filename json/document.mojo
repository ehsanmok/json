# json - Tape-backed Document representation.
#
# A `Document` is a single packed buffer of 64-bit tape entries plus a
# few side pools. Arrays and objects are read from the tape in O(1)
# per step: each container's header carries a `child_start_idx` that
# points backwards into a contiguous run of children's headers, so a
# linear walk over the tape is enough to traverse the entire DOM.
#
# Tape entry layout (64 bits)
# ---------------------------
# Bits 60-63: 4-bit type tag.
# Bits 0-59 : 60-bit payload (interpretation depends on tag).
#
# Tag layouts:
#   NULL    (0): payload unused.
#   BOOL    (1): payload low bit = 0/1.
#   INT     (2): payload = sign-extended 60-bit signed integer
#                (callers must ensure |value| < 2^59; values that overflow
#                are spilled into Document.int_pool in later phases).
#   FLOAT   (3): payload = 30-bit offset into Document.float_pool.
#   STRING  (4): payload = (30-bit input offset || 30-bit length).
#   ARRAY   (5): payload = (30-bit element_count || 30-bit child_start_idx).
#   OBJECT  (6): payload = (30-bit pair_count || 30-bit child_start_idx).
#
# For an OBJECT, the children at child_start_idx are stored as alternating
# (key, value) entries: the key entry is a STRING whose payload is an offset
# into Document.key_pool (rather than into Document.input), so unescaped
# keys can be cached. The value entry sits at child_start_idx + 1, and the
# next pair starts at child_start_idx + 2, etc.
#
# For an ARRAY, the children at child_start_idx are stored as a contiguous
# run of entries, one per element.

from std.collections import List
from std.memory import bitcast


# ---------------------------------------------------------------------------
# Tag constants
# ---------------------------------------------------------------------------

comptime TAPE_TAG_NULL: UInt8 = 0
comptime TAPE_TAG_BOOL: UInt8 = 1
comptime TAPE_TAG_INT: UInt8 = 2
comptime TAPE_TAG_FLOAT: UInt8 = 3
comptime TAPE_TAG_STRING: UInt8 = 4
comptime TAPE_TAG_ARRAY: UInt8 = 5
comptime TAPE_TAG_OBJECT: UInt8 = 6
# Tag 7: object KEY entries that point into key_pool. Used when the key
# had to be unescaped at parse time, or when keys are produced by an
# upstream source (e.g. simdjson FFI, owned-tree materialization) that
# does not preserve the original input bytes.
comptime TAPE_TAG_KEY: UInt8 = 7
# Tag 8 is for STRING values that needed unescaping. They live in
# string_pool the same way KEY entries live in key_pool. Clean strings
# stay in TAPE_TAG_STRING and remain zero-copy slices into input.
comptime TAPE_TAG_STRING_OWNED: UInt8 = 8
# Tag 9: zero-copy object KEY entries that store a (offset, length) pair
# into `Document.input`, identical to TAPE_TAG_STRING but with a different
# tag so iterators can distinguish keys from values without a sidecar.
# This is the by-far-most-common shape on real-world JSON corpora -- most
# object keys never contain escapes -- so emitting this tag instead of
# TAPE_TAG_KEY removes the per-key heap allocation + key_pool append from
# stage 2's hot path.
comptime TAPE_TAG_KEY_INLINE: UInt8 = 9
# Tag 10: an integer that does not fit the 60-bit inline payload, held
# in `int_pool`. Inlining every integer in 60 bits silently sign-flipped
# anything at or above 2**59 -- `576460752303423488` read back negative
# -- which is well inside the range JSON documents actually carry.
comptime TAPE_TAG_INT_POOL: UInt8 = 10
# Tag 11: a magnitude above `Int64.MAX` that still fits `UInt64`, held in
# `int_pool` as a bit pattern. JSON puts no upper bound on an integer,
# and the alternative for this range was wrapping to a negative number.
comptime TAPE_TAG_UINT: UInt8 = 11


# ---------------------------------------------------------------------------
# Bit packing
# ---------------------------------------------------------------------------

comptime _PAYLOAD_MASK: UInt64 = (UInt64(1) << 60) - 1
comptime _INLINE_INT_LIMIT: Int64 = Int64(1) << 59
"""Half the inline payload range; outside it an integer spills to the pool."""
comptime _OFFSET_MASK: UInt64 = (UInt64(1) << 30) - 1


@always_inline
def pack_tape_entry(tag: UInt8, payload: UInt64) -> UInt64:
    """Pack a 4-bit tag with a 60-bit payload into one tape entry."""
    return (UInt64(tag) << 60) | (payload & _PAYLOAD_MASK)


@always_inline
def tape_tag(entry: UInt64) -> UInt8:
    """Extract the 4-bit type tag from a tape entry."""
    return UInt8((entry >> 60) & 0xF)


@always_inline
def tape_payload(entry: UInt64) -> UInt64:
    """Extract the 60-bit payload from a tape entry."""
    return entry & _PAYLOAD_MASK


@always_inline
def pack_pair(hi30: UInt64, lo30: UInt64) -> UInt64:
    """Pack two 30-bit unsigned values into a 60-bit payload."""
    return ((hi30 & _OFFSET_MASK) << 30) | (lo30 & _OFFSET_MASK)


@always_inline
def payload_hi30(payload: UInt64) -> Int:
    """Return the high 30 bits of a payload as an Int."""
    return Int((payload >> 30) & _OFFSET_MASK)


@always_inline
def payload_lo30(payload: UInt64) -> Int:
    """Return the low 30 bits of a payload as an Int."""
    return Int(payload & _OFFSET_MASK)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


struct Document(Copyable):
    """Owns the JSON input bytes and a packed tape of entries.

    A `Document` is the storage unit produced by parsers. A `Value` view
    references a tape entry inside a `Document` by index; when a `Value`
    is mutated, the document is materialized into an owned tree
    (`OwnedValue` in `value/owned.mojo`).

    Side pools:
      - `key_pool` stores unescaped object keys. Object KEY entries store a
        pool offset rather than an offset into `input`, so unescaped keys
        survive even when the original bytes are released.
      - `string_pool` stores unescaped STRING values that needed
        materialisation (e.g. escapes). Clean strings stay zero-copy
        as STRING (offset, length) entries into `input`.
      - `float_pool` stores `Float64` values, since 64-bit IEEE 754 doesn't
        fit in 60 bits.
      - `int_pool` holds integers too wide for the 60-bit inline
        payload, both signed (`TAPE_TAG_INT_POOL`) and unsigned
        (`TAPE_TAG_UINT`, stored as a bit pattern).
    """

    var input: String
    var tape: List[UInt64]
    var key_pool: List[String]
    var string_pool: List[String]
    var float_pool: List[Float64]
    var int_pool: List[Int64]

    def __init__(out self):
        self.input = String()
        self.tape = List[UInt64]()
        self.key_pool = List[String]()
        self.string_pool = List[String]()
        self.float_pool = List[Float64]()
        self.int_pool = List[Int64]()

    def __init__(out self, var input: String):
        self.input = input^
        self.tape = List[UInt64]()
        self.key_pool = List[String]()
        self.string_pool = List[String]()
        self.float_pool = List[Float64]()
        self.int_pool = List[Int64]()

    def copy(self) -> Self:
        var d = Self()
        d.input = self.input
        d.tape = self.tape.copy()
        d.key_pool = self.key_pool.copy()
        d.string_pool = self.string_pool.copy()
        d.float_pool = self.float_pool.copy()
        d.int_pool = self.int_pool.copy()
        return d^

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def append_null(mut self) -> Int:
        var idx = len(self.tape)
        self.tape.append(pack_tape_entry(TAPE_TAG_NULL, 0))
        return idx

    def append_bool(mut self, b: Bool) -> Int:
        var idx = len(self.tape)
        var payload: UInt64 = 1 if b else 0
        self.tape.append(pack_tape_entry(TAPE_TAG_BOOL, payload))
        return idx

    def append_int(mut self, value: Int64) -> Int:
        """Append an integer, inline when it fits and pooled when not.

        The inline payload is 60 bits, so anything outside
        [-2**59, 2**59) has to spill. It used to be truncated instead,
        which turned `576460752303423488` into its negative -- no
        error, no warning, a different number.
        """
        var idx = len(self.tape)
        if value >= -_INLINE_INT_LIMIT and value < _INLINE_INT_LIMIT:
            var payload = UInt64(value) & _PAYLOAD_MASK
            self.tape.append(pack_tape_entry(TAPE_TAG_INT, payload))
            return idx
        var pool_idx = len(self.int_pool)
        self.int_pool.append(value)
        self.tape.append(pack_tape_entry(TAPE_TAG_INT_POOL, UInt64(pool_idx)))
        return idx

    def append_uint(mut self, value: UInt64) -> Int:
        """Append a magnitude above `Int64.MAX`.

        Stored as a bit pattern in the same pool; the tag is what says
        to read it back unsigned.
        """
        var idx = len(self.tape)
        var pool_idx = len(self.int_pool)
        self.int_pool.append(Int64(bitcast[DType.int64](value)))
        self.tape.append(pack_tape_entry(TAPE_TAG_UINT, UInt64(pool_idx)))
        return idx

    def append_float(mut self, value: Float64) -> Int:
        var pool_idx = len(self.float_pool)
        self.float_pool.append(value)
        var idx = len(self.tape)
        self.tape.append(pack_tape_entry(TAPE_TAG_FLOAT, UInt64(pool_idx)))
        return idx

    def append_string(mut self, offset: Int, length: Int) -> Int:
        """Append a STRING entry referencing bytes [offset:offset+length] of input.
        """
        var idx = len(self.tape)
        self.tape.append(
            pack_tape_entry(
                TAPE_TAG_STRING,
                pack_pair(UInt64(offset), UInt64(length)),
            )
        )
        return idx

    def append_key(mut self, var key: String) -> Int:
        """Intern a key into key_pool and append a KEY entry referencing it."""
        var pool_idx = len(self.key_pool)
        self.key_pool.append(key^)
        var idx = len(self.tape)
        self.tape.append(pack_tape_entry(TAPE_TAG_KEY, UInt64(pool_idx)))
        return idx

    def append_key_inline(mut self, offset: Int, length: Int) -> Int:
        """Append a zero-copy KEY entry referencing input[offset:offset+length].

        Used for the by-far-common case where the JSON key contains no
        escapes -- avoids the per-key heap allocation, memcpy, and
        key_pool append.
        """
        var idx = len(self.tape)
        self.tape.append(
            pack_tape_entry(
                TAPE_TAG_KEY_INLINE,
                pack_pair(UInt64(offset), UInt64(length)),
            )
        )
        return idx

    def append_string_owned(mut self, var value: String) -> Int:
        """Append a STRING_OWNED entry whose bytes live in string_pool.

        Used for JSON strings that contain escapes (where a zero-copy
        slice into `input` would expose the escaped bytes). Clean
        strings stay on the zero-copy `append_string` path.
        """
        var pool_idx = len(self.string_pool)
        self.string_pool.append(value^)
        var idx = len(self.tape)
        self.tape.append(
            pack_tape_entry(TAPE_TAG_STRING_OWNED, UInt64(pool_idx))
        )
        return idx

    def append_array(mut self, count: Int, child_start_idx: Int) -> Int:
        var idx = len(self.tape)
        self.tape.append(
            pack_tape_entry(
                TAPE_TAG_ARRAY,
                pack_pair(UInt64(count), UInt64(child_start_idx)),
            )
        )
        return idx

    def append_object(mut self, pair_count: Int, child_start_idx: Int) -> Int:
        var idx = len(self.tape)
        self.tape.append(
            pack_tape_entry(
                TAPE_TAG_OBJECT,
                pack_pair(UInt64(pair_count), UInt64(child_start_idx)),
            )
        )
        return idx

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    @always_inline
    def get_tag(self, tape_idx: Int) -> UInt8:
        return tape_tag(self.tape[tape_idx])

    @always_inline
    def get_payload(self, tape_idx: Int) -> UInt64:
        return tape_payload(self.tape[tape_idx])

    def get_bool(self, tape_idx: Int) -> Bool:
        return self.get_payload(tape_idx) == 1

    def get_int(self, tape_idx: Int) -> Int64:
        """Read an integer entry, inline or pooled."""
        if self.get_tag(tape_idx) == TAPE_TAG_INT_POOL:
            return self.int_pool[Int(self.get_payload(tape_idx))]
        if self.get_tag(tape_idx) == TAPE_TAG_UINT:
            return self.int_pool[Int(self.get_payload(tape_idx))]
        var payload = self.get_payload(tape_idx)
        # Sign-extend from 60 bits.
        var sign_bit = (payload >> 59) & 1
        if sign_bit == 1:
            return Int64(payload | (UInt64(0xF) << 60))
        return Int64(payload)

    def get_uint(self, tape_idx: Int) -> UInt64:
        """Read an entry as an unsigned magnitude.

        A `TAPE_TAG_UINT` entry is one that only fits here; the signed
        tags are accepted too so a caller can ask this question of any
        non-negative integer without checking the tag first.
        """
        if self.get_tag(tape_idx) == TAPE_TAG_UINT:
            return bitcast[DType.uint64](
                self.int_pool[Int(self.get_payload(tape_idx))]
            )
        return bitcast[DType.uint64](self.get_int(tape_idx))

    def get_float(self, tape_idx: Int) -> Float64:
        var pool_idx = Int(self.get_payload(tape_idx))
        return self.float_pool[pool_idx]

    def get_string_offset(self, tape_idx: Int) -> Int:
        return payload_hi30(self.get_payload(tape_idx))

    def get_string_length(self, tape_idx: Int) -> Int:
        return payload_lo30(self.get_payload(tape_idx))

    def get_string(self, tape_idx: Int) -> String:
        """Materialize a STRING / STRING_OWNED entry as a fresh `String`.

        - For `TAPE_TAG_STRING` (clean / zero-copy), the returned string
          is a fresh copy of the slice of `input` referenced by the
          (offset, length) payload. Safe even if `input` is later
          consumed.
        - For `TAPE_TAG_STRING_OWNED` (had escapes at parse time), the
          string is read out of `string_pool` and returned as-is.
        """
        var tag = self.get_tag(tape_idx)
        if tag == TAPE_TAG_STRING_OWNED:
            var pool_idx = Int(self.get_payload(tape_idx))
            return self.string_pool[pool_idx]
        var offset = self.get_string_offset(tape_idx)
        var length = self.get_string_length(tape_idx)
        return String(
            unsafe_from_utf8=self.input.as_bytes()[offset : offset + length]
        )

    def get_key(self, tape_idx: Int) -> String:
        """Return the unescaped key for a KEY / KEY_INLINE entry.

        Dispatches on the tag: TAPE_TAG_KEY is a key_pool index (used
        when the key had escapes, or when an upstream source produced
        the key as an owned string); TAPE_TAG_KEY_INLINE is a
        zero-copy `(offset, length)` slice into `input` that we copy
        out on read.
        """
        var tag = self.get_tag(tape_idx)
        if tag == TAPE_TAG_KEY_INLINE:
            var offset = self.get_string_offset(tape_idx)
            var length = self.get_string_length(tape_idx)
            return String(
                unsafe_from_utf8=self.input.as_bytes()[offset : offset + length]
            )
        var pool_idx = Int(self.get_payload(tape_idx))
        return self.key_pool[pool_idx]

    def get_count(self, tape_idx: Int) -> Int:
        """Element count for ARRAY, pair count for OBJECT."""
        return payload_hi30(self.get_payload(tape_idx))

    def get_child_start(self, tape_idx: Int) -> Int:
        """Tape index where this container's children start."""
        return payload_lo30(self.get_payload(tape_idx))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def root(self) -> Int:
        """Tape index of the root entry. Defined as the LAST entry, since
        builders write children before their parent so the parent can record
        a backwards-pointing child_start_idx."""
        return len(self.tape) - 1

    def size(self) -> Int:
        return len(self.tape)
