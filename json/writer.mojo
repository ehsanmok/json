# json - JsonWriter: the byte-level serialization sink.
#
# Every serialization path in the library emits through this type. It
# exists because the previous write path had no buffer at all: each node
# returned a fresh `String` that its parent concatenated, so a leaf's
# bytes were copied once per ancestor level, and string escaping built a
# heap `String` per input byte. Serializing a 47 KB document ran at
# ~110 MB/s as a result.
#
# The shape is the one both fast Mojo JSON libraries converge on:
#
#   * One `List[Byte]` whose *length* is set upfront with
#     `unsafe_uninit_length`, plus a `pos` cursor. This is not an
#     append-with-growth list -- `ensure()` is a cold escape hatch for
#     when the caller's size estimate was short, not the growth policy.
#   * `unsafe_memcpy` for spans, unaligned word stores for the fixed
#     literals (`null` / `true` / `false`).
#   * A SIMD pass to decide whether a string needs escaping at all;
#     the common answer is no, and then the whole string is one memcpy
#     between two quote bytes.
#   * Integers written backwards, two digits per step, out of a
#     100-entry pair table.
#   * `finish()` moves the buffer out. No copy.
#
# Using raw pointer stores rather than bounds-checked `List` indexing
# also sidesteps the Mojo stdlib's `ASSERT=safe` bounds checks, which
# `docs/performance.md` measures at ~20-37% on these workloads and which
# a consumer running plain `mojo run` cannot switch off.

from std.bit import count_trailing_zeros
from std.collections import List
from std.memory import bitcast, unsafe_memcpy
from std.memory.unsafe import pack_bits
from std.sys.info import simd_width_of
from std.utils.numerics import isfinite

from .dtoa import shortest_digits


comptime _SCAN_W = simd_width_of[DType.uint8]()

# Two ASCII digits per entry, so one 2-byte vector load yields both.
comptime _DIGIT_PAIRS = _make_digit_pairs()
comptime _HEX = "0123456789abcdef".as_bytes()

comptime _QUOTE = UInt8(0x22)
comptime _BACKSLASH = UInt8(0x5C)
comptime _SPACE = UInt8(0x20)


def _make_digit_pairs(out s: InlineArray[SIMD[DType.uint8, 2], 100]):
    s = InlineArray[SIMD[DType.uint8, 2], 100](uninitialized=True)
    for i in range(100):
        s[i] = SIMD[DType.uint8, 2](
            UInt8(0x30 + (i // 10)), UInt8(0x30 + (i % 10))
        )


def _int_digits(v: UInt64) -> Int:
    """Decimal digit count. Unrolled compare ladder, no division."""
    if v < 10:
        return 1
    if v < 100:
        return 2
    if v < 1_000:
        return 3
    if v < 10_000:
        return 4
    if v < 100_000:
        return 5
    if v < 1_000_000:
        return 6
    if v < 10_000_000:
        return 7
    if v < 100_000_000:
        return 8
    if v < 1_000_000_000:
        return 9
    if v < 10_000_000_000:
        return 10
    var n = 10
    var x = v
    while x >= 10_000_000_000:
        x //= 10
        n += 1
    return n


def needs_escape(b: Span[UInt8, _]) -> Bool:
    """True if any byte requires escaping in a JSON string literal.

    One SIMD pass testing `== '"'`, `== '\\'` and `< 0x20` per byte. The
    answer is almost always False, which is what makes the memcpy fast
    path in `write_string` worth having.
    """
    var n = len(b)
    var ptr = b.unsafe_ptr()
    var i = 0
    var q = SIMD[DType.uint8, _SCAN_W](_QUOTE)
    var bs = SIMD[DType.uint8, _SCAN_W](_BACKSLASH)
    var sp = SIMD[DType.uint8, _SCAN_W](_SPACE)
    while i + _SCAN_W <= n:
        var chunk = ptr.unsafe_load[width=_SCAN_W](i)
        var hit = chunk.eq(q) | chunk.eq(bs) | chunk.lt(sp)
        if Int(pack_bits(hit)) != 0:
            return True
        i += _SCAN_W
    while i < n:
        var c = b[i]
        if c == _QUOTE or c == _BACKSLASH or c < _SPACE:
            return True
        i += 1
    return False


struct JsonWriter(Movable):
    """Append-only JSON byte sink over a pre-sized buffer.

    Construct with a capacity estimate, write, then `finish()`. The
    estimate does not have to be right -- `ensure` grows the buffer when
    it is short -- but a good estimate means the hot path never checks
    anything but the cursor.

    Example:
        var w = JsonWriter(capacity=64)
        w.begin_object()
        w.key("name")
        w.write_string("Ada")
        w.end_object()
        var out = w^.finish_string()
    """

    var buf: List[UInt8]
    var pos: Int
    # Non-empty enables pretty output: containers break lines and nest
    # by `indent` per level. Empty (the default) is compact.
    var indent: String
    var depth: Int
    # True once the current container has at least one member, so the
    # next `key`/value knows whether to emit a leading comma. Stacked in
    # `_depth_flags`, one bit per open container.
    var _has_member: Bool
    var _depth_flags: List[Bool]

    def __init__(out self, *, capacity: Int = 256):
        if capacity > 0:
            self.buf = List[UInt8](unsafe_uninit_length=capacity)
        else:
            self.buf = List[UInt8]()
        self.pos = 0
        self.indent = String()
        self.depth = 0
        self._has_member = False
        self._depth_flags = List[Bool]()

    def __init__(out self, *, capacity: Int = 256, indent: String):
        """Pretty-printing writer. `indent` is one level of indentation."""
        if capacity > 0:
            self.buf = List[UInt8](unsafe_uninit_length=capacity)
        else:
            self.buf = List[UInt8]()
        self.pos = 0
        self.indent = indent
        self.depth = 0
        self._has_member = False
        self._depth_flags = List[Bool]()

    def __init__(out self, var buf: List[UInt8]):
        """Adopt a caller-owned buffer, so allocations can be reused."""
        self.buf = buf^
        self.pos = 0
        self.indent = String()
        self.depth = 0
        self._has_member = False
        self._depth_flags = List[Bool]()

    @always_inline
    def ensure(mut self, n: Int):
        """Make room for `n` more bytes. Cold path when sized correctly."""
        var need = self.pos + n
        if need > len(self.buf):
            var grown = need * 2
            self.buf.resize(unsafe_uninit_length=grown)

    @always_inline
    def _put(mut self, c: UInt8):
        self.buf.unsafe_ptr().unsafe_offset(self.pos)[] = c
        self.pos += 1

    @always_inline
    def write_byte(mut self, c: UInt8):
        self.ensure(1)
        self._put(c)

    def write_bytes(mut self, data: Span[UInt8, _]):
        var n = len(data)
        if n == 0:
            return
        self.ensure(n)
        unsafe_memcpy(
            dest=self.buf.unsafe_ptr().unsafe_offset(self.pos),
            src=data.unsafe_ptr(),
            count=n,
        )
        self.pos += n

    @always_inline
    def write_literal(mut self, s: StaticString):
        self.write_bytes(s.as_bytes())

    def reset(mut self):
        """Rewind to empty, keeping the buffer.

        For a caller serializing many values in a row: the allocation
        is the expensive part and it is already the right size.
        """
        self.pos = 0

    # --- scalars -----------------------------------------------------

    def write_null(mut self):
        self.write_literal("null")

    def write_bool(mut self, b: Bool):
        if b:
            self.write_literal("true")
        else:
            self.write_literal("false")

    def write_int(mut self, v: Int64):
        """Decimal integer, written backwards two digits at a time.

        The magnitude is taken in unsigned arithmetic, so
        `Int64.MIN` -- which has no positive counterpart -- needs no
        special case: negating it as `UInt64` gives exactly 2**63.
        """
        if v == 0:
            self.write_byte(UInt8(0x30))
            return
        var neg = v < 0
        var mag = (UInt64(0) - UInt64(v)) if neg else UInt64(v)
        self._write_digits(mag, neg)

    def write_uint(mut self, v: UInt64):
        """Decimal integer above the signed range.

        JSON puts no upper bound on an integer, so a document may hold
        a value that only fits unsigned. Writing it through the signed
        path would wrap it into a negative number, silently.
        """
        if v == 0:
            self.write_byte(UInt8(0x30))
            return
        self._write_digits(v, False)

    @always_inline
    def _write_digits(mut self, mag: UInt64, negative: Bool):
        var digits = _int_digits(mag)
        self.ensure(digits + 1)
        if negative:
            self._put(UInt8(0x2D))
        var start = self.pos
        var write = start + digits
        var x = mag
        var base = self.buf.unsafe_ptr()
        var pairs = materialize[_DIGIT_PAIRS]()
        while x >= 100:
            var r = Int(x % 100)
            x //= 100
            write -= 2
            var pair = pairs[r]
            base.unsafe_offset(write)[] = pair[0]
            base.unsafe_offset(write + 1)[] = pair[1]
        if x >= 10:
            write -= 2
            var pair = pairs[Int(x)]
            base.unsafe_offset(write)[] = pair[0]
            base.unsafe_offset(write + 1)[] = pair[1]
        else:
            write -= 1
            base.unsafe_offset(write)[] = UInt8(0x30 + Int(x))
        self.pos = start + digits

    def write_float(mut self, v: Float64):
        """A float as the shortest decimal that reads back as itself.

        Digits come from `json.dtoa`, not from `String(Float64)`: the
        stdlib formatter does not always produce a representation that
        round-trips, and a serializer that writes a number nothing can
        read back as the value given to it has lost the one property
        it exists to keep.

        Layout follows the convention this library already emitted --
        fixed notation while the leading digit sits between 1e-5 and
        1e15, scientific outside that, exponent always signed and at
        least two digits. Integer-valued floats keep a trailing `.0`
        so that a value written as a float reads back as one.

        A non-finite value writes `null`. JSON has no spelling for
        infinity or NaN, so the alternatives are an invalid document
        or a refusal; refusing needs a policy the caller sets, which
        `SerializerConfig` grows separately.
        """
        if not isfinite(v):
            self.write_null()
            return

        self.ensure(32)
        var bits = bitcast[DType.uint64](v)
        if (bits >> 63) != 0:
            self._put(UInt8(0x2D))
        var magnitude = bitcast[DType.float64](bits & (UInt64.MAX >> 1))

        if magnitude == 0.0:
            self._put(UInt8(0x30))
            self._put(UInt8(0x2E))
            self._put(UInt8(0x30))
            return

        var digits = InlineArray[UInt8, 24](uninitialized=True)
        var generated = shortest_digits(magnitude, digits)
        var count = generated[0]
        var exponent = generated[1]
        # Decimal exponent of the leading digit.
        var leading = exponent + count - 1

        if leading < -4 or leading > 15:
            self._write_scientific(digits, count, leading)
        elif exponent >= 0:
            # An integer: digits, then the zeros the exponent implies.
            for i in range(count):
                self._put(digits[i])
            for _ in range(exponent):
                self._put(UInt8(0x30))
            self._put(UInt8(0x2E))
            self._put(UInt8(0x30))
        else:
            var integer_digits = count + exponent
            if integer_digits <= 0:
                self._put(UInt8(0x30))
                self._put(UInt8(0x2E))
                for _ in range(-integer_digits):
                    self._put(UInt8(0x30))
                for i in range(count):
                    self._put(digits[i])
            else:
                for i in range(integer_digits):
                    self._put(digits[i])
                self._put(UInt8(0x2E))
                for i in range(integer_digits, count):
                    self._put(digits[i])

    def _write_scientific(
        mut self, digits: InlineArray[UInt8, 24], count: Int, leading: Int
    ):
        """`d.ddde±NN`, with at least two exponent digits."""
        self._put(digits[0])
        if count > 1:
            self._put(UInt8(0x2E))
            for i in range(1, count):
                self._put(digits[i])
        self._put(UInt8(0x65))
        var exponent = leading
        if exponent < 0:
            self._put(UInt8(0x2D))
            exponent = -exponent
        else:
            self._put(UInt8(0x2B))
        if exponent < 10:
            self._put(UInt8(0x30))
            self._put(UInt8(0x30 + exponent))
        elif exponent < 100:
            var pairs = materialize[_DIGIT_PAIRS]()
            var pair = pairs[exponent]
            self._put(pair[0])
            self._put(pair[1])
        else:
            var hundreds = exponent // 100
            self._put(UInt8(0x30 + hundreds))
            var pairs = materialize[_DIGIT_PAIRS]()
            var pair = pairs[exponent % 100]
            self._put(pair[0])
            self._put(pair[1])

    def write_string(mut self, s: String):
        """A quoted, escaped JSON string literal."""
        self.write_string_span(s.as_bytes())

    def write_string_span(mut self, b: Span[UInt8, _]):
        """A quoted, escaped JSON string literal.

        One pass, not two. The scan that decides whether anything
        needs escaping used to be separate from the copy, so every
        string was read twice even though the scan already knows
        where the clean run ends. Here the first chunk with something
        to escape hands straight over to the escape path, which
        restarts from the beginning; a string with no escapes -- the
        usual case -- is a quote, one memcpy and a quote.
        """
        var n = len(b)
        if needs_escape(b):
            self._write_string_escaped(b)
            return
        self.ensure(n + 2)
        self._put(_QUOTE)
        if n > 0:
            unsafe_memcpy(
                dest=self.buf.unsafe_ptr().unsafe_offset(self.pos),
                src=b.unsafe_ptr(),
                count=n,
            )
            self.pos += n
        self._put(_QUOTE)

    def _write_string_escaped(mut self, b: Span[UInt8, _]):
        """Escape path: bulk-copy the clean runs between escapes.

        Worst case is 6 bytes out per byte in, plus the two quotes.
        """
        var n = len(b)
        self.ensure(n * 6 + 2)
        self._put(_QUOTE)
        var ptr = b.unsafe_ptr()
        var i = 0
        var start = 0
        var q = SIMD[DType.uint8, _SCAN_W](_QUOTE)
        var bs = SIMD[DType.uint8, _SCAN_W](_BACKSLASH)
        var sp = SIMD[DType.uint8, _SCAN_W](_SPACE)

        while i + _SCAN_W <= n:
            var chunk = ptr.unsafe_load[width=_SCAN_W](i)
            var bits = pack_bits(chunk.eq(q) | chunk.eq(bs) | chunk.lt(sp))
            if Int(bits) == 0:
                i += _SCAN_W
                continue
            var m = bits
            while Int(m) != 0:
                var off = Int(count_trailing_zeros(m))
                var at = i + off
                if at > start:
                    self.write_bytes(b[start:at])
                self._escape_one(b[at])
                start = at + 1
                m &= m - 1
            i += _SCAN_W

        while i < n:
            var c = b[i]
            if c == _QUOTE or c == _BACKSLASH or c < _SPACE:
                if i > start:
                    self.write_bytes(b[start:i])
                self._escape_one(c)
                start = i + 1
            i += 1

        if start < n:
            self.write_bytes(b[start:n])
        self.write_byte(_QUOTE)

    def _escape_one(mut self, c: UInt8):
        """Emit the escape sequence for one byte that needs it."""
        self.ensure(6)
        if c == _QUOTE:
            self._put(_BACKSLASH)
            self._put(_QUOTE)
        elif c == _BACKSLASH:
            self._put(_BACKSLASH)
            self._put(_BACKSLASH)
        elif c == UInt8(0x0A):
            self._put(_BACKSLASH)
            self._put(UInt8(0x6E))  # n
        elif c == UInt8(0x0D):
            self._put(_BACKSLASH)
            self._put(UInt8(0x72))  # r
        elif c == UInt8(0x09):
            self._put(_BACKSLASH)
            self._put(UInt8(0x74))  # t
        elif c == UInt8(0x08):
            self._put(_BACKSLASH)
            self._put(UInt8(0x62))  # b
        elif c == UInt8(0x0C):
            self._put(_BACKSLASH)
            self._put(UInt8(0x66))  # f
        else:
            # Remaining C0 controls: six-character hex escape.
            self._put(_BACKSLASH)
            self._put(UInt8(0x75))  # u
            self._put(UInt8(0x30))
            self._put(UInt8(0x30))
            self._put(_HEX[Int(c) >> 4])
            self._put(_HEX[Int(c) & 0xF])

    # --- structure ---------------------------------------------------
    #
    # The comma bookkeeping lives here so callers cannot get it wrong,
    # which is the single most common hand-rolled-serializer bug.

    def _sep(mut self):
        if self._has_member:
            self.write_byte(UInt8(0x2C))
        self._has_member = True

    def begin_object(mut self):
        self._sep()
        self._depth_flags.append(self._has_member)
        self._has_member = False
        self.write_byte(UInt8(0x7B))

    def end_object(mut self):
        self.write_byte(UInt8(0x7D))
        self._has_member = (
            self._depth_flags.pop() if len(self._depth_flags) > 0 else True
        )

    def begin_array(mut self):
        self._sep()
        self._depth_flags.append(self._has_member)
        self._has_member = False
        self.write_byte(UInt8(0x5B))

    def end_array(mut self):
        self.write_byte(UInt8(0x5D))
        self._has_member = (
            self._depth_flags.pop() if len(self._depth_flags) > 0 else True
        )

    def key(mut self, k: String):
        """An object key plus its colon."""
        if self._has_member:
            self.write_byte(UInt8(0x2C))
        self._has_member = True
        self.write_string(k)
        self.write_byte(UInt8(0x3A))
        # A key is not itself a member for separator purposes; the value
        # that follows must not emit a comma before itself.
        self._has_member = False

    def value_written(mut self):
        """Mark that a value completed, so the next member emits a comma."""
        self._has_member = True

    # --- values in a container ---------------------------------------
    #
    # These are the ones callers should reach for: they handle the
    # separator, so a container body is a flat sequence of calls.

    def item_null(mut self):
        self._sep()
        self.write_null()

    def item_bool(mut self, b: Bool):
        self._sep()
        self.write_bool(b)

    def item_int(mut self, v: Int64):
        self._sep()
        self.write_int(v)

    def item_float(mut self, v: Float64):
        self._sep()
        self.write_float(v)

    def item_string(mut self, s: String):
        self._sep()
        self.write_string(s)

    # --- structural helpers for the tree walkers ---------------------
    #
    # `dumps` compact and `dumps(indent=...)` differ only in whitespace,
    # so both walkers call these and neither knows which mode it is in.
    # Pretty output used to be produced by serializing the whole document
    # compactly and then re-scanning the result byte-at-a-time.

    @always_inline
    def _newline(mut self):
        var unit = self.indent.byte_length()
        if unit == 0:
            return
        # Read the indent through a raw pointer: a `Span` borrowed from
        # `self.indent` would alias the `mut self` the writes need.
        var src = self.indent.unsafe_ptr()
        self.ensure(1 + self.depth * unit)
        self._put(UInt8(0x0A))
        var dest = self.buf.unsafe_ptr()
        for _ in range(self.depth):
            unsafe_memcpy(
                dest=dest.unsafe_offset(self.pos), src=src, count=unit
            )
            self.pos += unit

    def open_container(mut self, brace: UInt8):
        self.write_byte(brace)
        self.depth += 1

    def close_container(mut self, brace: UInt8, empty: Bool):
        self.depth -= 1
        if not empty:
            self._newline()
        self.write_byte(brace)

    def next_child(mut self, first: Bool):
        """Separator before a container child."""
        if not first:
            self.write_byte(UInt8(0x2C))
        self._newline()

    def colon(mut self):
        self.write_byte(UInt8(0x3A))
        if self.indent.byte_length() > 0:
            self.write_byte(UInt8(0x20))

    # --- finishing ---------------------------------------------------

    def finish(deinit self) -> List[UInt8]:
        """The written bytes. Moves the buffer out; no copy."""
        if self.pos < len(self.buf):
            self.buf.resize(unsafe_uninit_length=self.pos)
        return self.buf^

    def finish_string(deinit self) -> String:
        """The written bytes as a `String`."""
        if self.pos < len(self.buf):
            self.buf.resize(unsafe_uninit_length=self.pos)
        return String(unsafe_from_utf8=self.buf)
