# A byte cursor for reading JSON straight into typed values.
#
# The typed read path used to go `loads` -> tape `Document` -> `Value`
# walk -> struct. That builds a whole document representation, with its
# own allocations and refcounts, only to throw it away one field later:
# a hundred records cost a tape, a string pool, an atomic increment per
# field access, and a `String` allocation per object key merely to
# compare it. And it could not build a `List[<struct>]` at all.
#
# This reads forward through the bytes and writes each value into its
# final home. No tape, no `Value`, no intermediate anything. It is used
# by `deserialize_json`; `loads` still exists and still produces a
# `Document`, because a caller who wants to navigate a document rather
# than decode it into known types needs exactly that.
#
# Validation is not relaxed for speed. The reader calls the same
# `scan_number` and the same string validators stage 2 does, so a
# document the tape parser rejects is rejected here, byte for byte,
# with the same message.

from std.bit import count_trailing_zeros
from std.collections import List
from std.memory.unsafe import pack_bits
from std.sys import simd_byte_width

from .cpu.number_parse import (
    NUM_ERR_LEADING_ZERO,
    NUM_FLOAT,
    NUM_INVALID,
    NUM_UINT,
    parse_float_span,
    parse_int_checked,
    scan_number,
)
from .cpu.validate import (
    ESC_BAD_CHAR,
    ESC_OK,
    ESC_TRAILING_BACKSLASH,
    STR_CONTROL,
    STR_ESCAPE,
    STR_NON_ASCII,
    find_control_char,
    first_invalid_utf8,
    is_valid_utf8,
    is_ws,
    skip_ws,
    validate_escapes,
)
from .errors import parse_error
from .unicode import unescape_json_string_span

comptime _QUOTE = UInt8(0x22)
comptime _BACKSLASH = UInt8(0x5C)
comptime _SPACE = UInt8(0x20)
comptime _BLOCK: Int = simd_byte_width()
"""One native SIMD register, so a chunk costs one instruction."""
comptime _MASK = _mask_dtype()
"""An unsigned integer wide enough to hold one bit per byte of a block."""


def _mask_dtype() -> DType:
    """One bit per byte of a SIMD block, as an integer type."""
    comptime assert (
        _BLOCK == 16 or _BLOCK == 32 or _BLOCK == 64
    ), "unsupported simd_byte_width()"
    comptime if _BLOCK == 16:
        return DType.uint16
    elif _BLOCK == 32:
        return DType.uint32
    else:
        return DType.uint64


comptime DEFAULT_MAX_DEPTH = 1024
"""Nesting past this is refused. Deep input is otherwise a stack hazard."""


@always_inline
def _narrow[
    dtype: DType
](magnitude: UInt64, negative: Bool) raises -> Scalar[dtype]:
    """Fit an already-validated magnitude into `dtype`, or raise.

    The wording matches `parse_int_checked`, because which of the two
    paths read the number is not something a caller should be able to
    tell from the error.
    """
    comptime if dtype.is_signed():
        if negative:
            if magnitude > UInt64(Scalar[dtype].MAX_FINITE) + 1:
                raise Error("integer out of range for " + String(dtype))
            if magnitude == UInt64(Scalar[dtype].MAX_FINITE) + 1:
                return Scalar[dtype].MIN_FINITE
            return -Scalar[dtype](magnitude)
        if magnitude > UInt64(Scalar[dtype].MAX_FINITE):
            raise Error("integer out of range for " + String(dtype))
        return Scalar[dtype](magnitude)
    else:
        if negative:
            raise Error("negative integer in an unsigned field")
        if magnitude > UInt64(Scalar[dtype].MAX_FINITE):
            raise Error("integer out of range for " + String(dtype))
        return Scalar[dtype](magnitude)


@fieldwise_init
struct KeySpan(ImplicitlyCopyable):
    """Where an object key's bytes are, and whether they need expanding."""

    var start: Int
    var end: Int
    var escaped: Bool


struct JsonReader[origin: ImmOrigin](Movable):
    """A forward cursor over JSON bytes.

    Borrows its input rather than owning it, so decoding a document
    does not begin by copying it.
    """

    var data: Span[UInt8, Self.origin]
    var pos: Int
    var depth: Int
    var max_depth: Int
    var scratch: List[UInt8]
    """Reused buffer for keys and strings that contain escapes."""

    def __init__(
        out self,
        data: Span[UInt8, Self.origin],
        *,
        max_depth: Int = DEFAULT_MAX_DEPTH,
    ):
        self.data = data
        self.pos = 0
        self.depth = 0
        self.max_depth = max_depth
        self.scratch = List[UInt8]()
        # RFC 8259 section 8.1 allows a reader to ignore a byte order
        # mark, and enough producers emit one that refusing is not
        # useful behaviour.
        if (
            len(data) >= 3
            and data[0] == UInt8(0xEF)
            and data[1] == UInt8(0xBB)
            and data[2] == UInt8(0xBF)
        ):
            self.pos = 3

    # --- position ----------------------------------------------------

    def error(self, message: String) -> Error:
        """A located error at the cursor."""
        return parse_error(message, self.data, min(self.pos, len(self.data)))

    def error_at(self, message: String, at: Int) -> Error:
        return parse_error(message, self.data, at)

    @always_inline
    def skip_ws(mut self):
        """Advance past whitespace.

        A hand-written inline check for "the next byte is not
        whitespace" before this call was measured and was slower on
        every shape: the shared scan already opens with a scalar
        prelude for exactly that case, and the extra branch only
        added work.
        """
        self.pos = skip_ws(self.data, self.pos, len(self.data))

    @always_inline
    def peek(mut self) -> UInt8:
        """The next significant byte, or 0 at end of input.

        Zero is safe as the end marker: a NUL outside a string is not
        in the grammar, so every caller that compares against a real
        byte rejects it anyway.
        """
        self.skip_ws()
        if self.pos >= len(self.data):
            return 0
        return self.data[self.pos]

    def at_end(mut self) -> Bool:
        self.skip_ws()
        return self.pos >= len(self.data)

    def expect_end(mut self) raises:
        """Nothing but whitespace may follow the top-level value."""
        if not self.at_end():
            raise self.error("trailing content after top-level value")

    # --- containers --------------------------------------------------

    def _enter(mut self) raises:
        self.depth += 1
        if self.depth > self.max_depth:
            raise self.error("nesting depth exceeds " + String(self.max_depth))

    @always_inline
    def expect_object_begin(mut self) raises:
        if self.peek() != UInt8(ord("{")):
            raise self.error("expected '{'")
        self.pos += 1
        self._enter()

    @always_inline
    def expect_array_begin(mut self) raises:
        if self.peek() != UInt8(ord("[")):
            raise self.error("expected '['")
        self.pos += 1
        self._enter()

    @always_inline
    def next_member(mut self, first: Bool) raises -> Bool:
        """Advance to the next object member. False at the closing brace.

        Each byte is looked at once. The previous shape called `peek`
        three times per member, and every call re-ran the whitespace
        scan from a position where nothing had been consumed since the
        last one; the byte it returned was already in a register.
        """
        var c = self.peek()
        if c == UInt8(ord("}")):
            self.pos += 1
            self.depth -= 1
            return False
        if not first:
            if c != UInt8(ord(",")):
                raise self.error("expected ',' or '}' in object")
            self.pos += 1
            c = self.peek()
            if c == UInt8(ord("}")):
                raise self.error("trailing comma in object")
        if c != _QUOTE:
            raise self.error("expected string key")
        return True

    @always_inline
    def next_element(mut self, first: Bool) raises -> Bool:
        """Advance to the next array element. False at the closing bracket."""
        var c = self.peek()
        if c == UInt8(ord("]")):
            self.pos += 1
            self.depth -= 1
            return False
        if not first:
            if c != UInt8(ord(",")):
                raise self.error("expected ',' or ']' in array")
            self.pos += 1
            c = self.peek()
            if c == UInt8(ord("]")):
                raise self.error("trailing comma in array")
        # `peek` reports 0 both for a NUL byte and for end of input; only
        # the second is an unterminated array, and a NUL here would be
        # rejected as an unexpected character by whatever reads next.
        if c == 0 and self.pos >= len(self.data):
            raise self.error("unterminated array")
        return True

    # --- strings -----------------------------------------------------

    def _scan_string(self, start: Int) -> Tuple[Int, UInt8]:
        """Find the closing quote and classify the body in one pass.

        Returns the offset of the quote that ends the string, or -1 if
        there is none, together with the `STR_*` flags describing what
        the body holds.

        This used to be two passes: a vector scan for the quote, then
        `scan_string_body` over the same bytes for the flags. The
        second pass only vectorizes at sixteen bytes or more, and keys
        and values in real documents are shorter than that, so it
        degenerated to three compares per byte over bytes that had
        just been in a vector register.

        The masking is what makes one pass correct. Within the block
        that contains the closing quote, only the bytes *before* it
        belong to this string; without the mask a control character or
        a non-ASCII byte sitting after the quote -- the next key, the
        rest of the document -- would set a flag for a string that does
        not contain it, and the caller would reject a valid document.
        """
        var ptr = self.data.unsafe_ptr()
        var n = len(self.data)
        var i = start
        var flags: UInt8 = 0
        var stop = n - _BLOCK

        while i <= stop:
            var chunk = ptr.unsafe_load[width=_BLOCK](i)
            var hit = pack_bits[dtype=_MASK](
                chunk.eq(_QUOTE) | chunk.eq(_BACKSLASH)
            )
            var ctl = pack_bits[dtype=_MASK](chunk.lt(_SPACE))
            var non = pack_bits[dtype=_MASK](chunk.ge(UInt8(0x80)))
            if hit == 0:
                if ctl != 0:
                    flags |= STR_CONTROL
                if non != 0:
                    flags |= STR_NON_ASCII
                i += _BLOCK
                continue
            var off = Int(count_trailing_zeros(hit))
            var before = (Scalar[_MASK](1) << Scalar[_MASK](off)) - 1
            if (ctl & before) != 0:
                flags |= STR_CONTROL
            if (non & before) != 0:
                flags |= STR_NON_ASCII
            var at = i + off
            if ptr[unsafe_offset=at] == _QUOTE:
                return (at, flags)
            flags |= STR_ESCAPE
            i = at + 2

        while i < n:
            var c = ptr[unsafe_offset=i]
            if c == _QUOTE:
                return (i, flags)
            if c == _BACKSLASH:
                flags |= STR_ESCAPE
                i += 2
                continue
            if c < _SPACE:
                flags |= STR_CONTROL
            elif c >= UInt8(0x80):
                flags |= STR_NON_ASCII
            i += 1
        return (-1, flags)

    @always_inline
    def _string_bounds(mut self) raises -> Tuple[Int, Int, UInt8]:
        """Consume a string literal; return its body and scan flags."""
        if self.peek() != _QUOTE:
            raise self.error("expected a string")
        return self._string_bounds_at_quote()

    @always_inline
    def _string_bounds_at_quote(mut self) raises -> Tuple[Int, Int, UInt8]:
        """The same, for a caller that has already seen the quote.

        `next_member` has to look at the byte anyway to tell a key from
        a closing brace, so re-running the whitespace scan and
        re-reading the byte to confirm what it just proved is work
        nobody asked for.
        """
        var start = self.pos + 1
        var scanned = self._scan_string(start)
        var end = scanned[0]
        var flags = scanned[1]
        if end < 0:
            raise self.error_at("unterminated string", self.pos)
        self.pos = end + 1

        if flags & STR_CONTROL != 0:
            var at = find_control_char(self.data, start, end)
            raise self.error_at("control character in string", at)
        if flags & STR_NON_ASCII != 0:
            if not is_valid_utf8(self.data[start:end]):
                raise self.error_at(
                    "invalid UTF-8 in string",
                    first_invalid_utf8(self.data, start, end),
                )
        if flags & STR_ESCAPE != 0:
            var err_pos = start
            var code = validate_escapes(self.data, start, end, False, err_pos)
            if code == ESC_TRAILING_BACKSLASH:
                raise self.error_at("trailing backslash in string", err_pos)
            if code == ESC_BAD_CHAR:
                raise self.error_at("invalid escape", err_pos)
            if code != ESC_OK:
                raise self.error_at("invalid \\u escape", err_pos)
        return (start, end, flags)

    def read_string(mut self) raises -> String:
        var bounds = self._string_bounds()
        if bounds[2] & STR_ESCAPE == 0:
            return String(unsafe_from_utf8=self.data[bounds[0] : bounds[1]])
        var expanded = unescape_json_string_span(
            self.data, bounds[0], bounds[1]
        )
        return String(unsafe_from_utf8=expanded^)

    @always_inline
    def read_key(mut self) raises -> KeySpan:
        """Consume an object key and its colon, without materializing it.

        A key is compared against a field name far more often than it
        is kept, and building a `String` for each comparison is what
        made the old read path allocate on the order of the square of
        the field count per record.
        """
        var bounds = self._string_bounds_at_quote()
        if self.peek() != UInt8(ord(":")):
            raise self.error("expected ':' after object key")
        self.pos += 1
        return KeySpan(bounds[0], bounds[1], bounds[2] & STR_ESCAPE != 0)

    @always_inline
    def key_matches[name: StaticString](self, key: KeySpan) -> Bool:
        """Whether an unescaped key spells `name`.

        Takes `self` immutably and the name as a parameter, both for
        the same reason: this runs inside a comptime-unrolled loop over
        a struct's fields, and a method that could mutate the reader
        forces the compiler to reload its span and position after every
        call in that loop. With the name known at compile time the
        length test is a comparison against a constant, which settles
        most fields without looking at a byte.
        """
        comptime wanted = name.as_bytes()
        comptime width = len(wanted)
        if key.end - key.start != width:
            return False
        var ptr = self.data.unsafe_ptr()
        comptime for i in range(width):
            if ptr[unsafe_offset=key.start + i] != wanted[i]:
                return False
        return True

    def key_equals(mut self, key: KeySpan, name: StaticString) -> Bool:
        """Whether a key spells `name`, expanding escapes if it has any.

        The escaped case is why this takes `self` mutably: the key is
        unescaped into the reader's scratch buffer before comparing, so
        that a key written `"\u0061"` still matches a field named `a`.
        Callers on the hot path go through `key_matches` and only reach
        here when the key actually carries an escape.
        """
        var wanted = name.as_bytes()
        if not key.escaped:
            if key.end - key.start != len(wanted):
                return False
            var ptr = self.data.unsafe_ptr()
            for i in range(len(wanted)):
                if ptr[unsafe_offset=key.start + i] != wanted[i]:
                    return False
            return True
        self.scratch = unescape_json_string_span(self.data, key.start, key.end)
        if len(self.scratch) != len(wanted):
            return False
        for i in range(len(wanted)):
            if self.scratch[i] != wanted[i]:
                return False
        return True

    def key_text(mut self, key: KeySpan) raises -> String:
        """A key as a `String`, for the cases that must keep it."""
        if not key.escaped:
            return String(unsafe_from_utf8=self.data[key.start : key.end])
        var expanded = unescape_json_string_span(self.data, key.start, key.end)
        return String(unsafe_from_utf8=expanded^)

    # --- scalars -----------------------------------------------------

    @always_inline
    def read_bool(mut self) raises -> Bool:
        var c = self.peek()
        if c == UInt8(ord("t")):
            self._expect_literal("true")
            return True
        if c == UInt8(ord("f")):
            self._expect_literal("false")
            return False
        raise self.error("expected a boolean")

    def read_null(mut self) raises:
        self._expect_literal("null")

    @always_inline
    def try_null(mut self) raises -> Bool:
        """Consume `null` if that is what comes next."""
        if self.peek() != UInt8(ord("n")):
            return False
        self._expect_literal("null")
        return True

    def _expect_literal(mut self, literal: StaticString) raises:
        var bytes = literal.as_bytes()
        self.skip_ws()
        if self.pos + len(bytes) > len(self.data):
            raise self.error("expected '" + String(literal) + "' literal")
        for i in range(len(bytes)):
            if self.data[self.pos + i] != bytes[i]:
                raise self.error("expected '" + String(literal) + "' literal")
        self.pos += len(bytes)

    @always_inline
    def read_int[dtype: DType](mut self) raises -> Scalar[dtype]:
        """Read an integer field.

        Most integers in a document are a handful of digits, and
        reaching them through the full number scanner costs four
        outlined calls: the scanner walks the digits once to check the
        grammar and again to accumulate, peeks for an eight-digit SWAR
        block that a short integer never has, and hands back a
        forty-eight byte token that the range check then unpacks.

        The path below commits only when it has itself established the
        whole token: an optional minus, a leading digit in 1-9 so that
        anything starting `0` defers, at most eighteen digits so the
        accumulator cannot overflow, and a terminator that cannot
        continue a number. Everything else -- `0`, `-0`, a fraction, an
        exponent, a leading zero, nineteen digits or more -- falls
        through to the scanner, which stays the only authority on the
        grammar, the messages and the offsets they carry.
        """
        self.skip_ws()
        var n = len(self.data)
        var ptr = self.data.unsafe_ptr()
        var p = self.pos
        var negative = p < n and ptr[unsafe_offset=p] == UInt8(ord("-"))
        var start = p + Int(negative)
        if start < n:
            var lead = ptr[unsafe_offset=start]
            if lead > UInt8(ord("0")) and lead <= UInt8(ord("9")):
                var acc = UInt64(lead) - UInt64(ord("0"))
                var i = start + 1
                while i < n and i - start < 18:
                    var c = ptr[unsafe_offset=i]
                    if c < UInt8(ord("0")) or c > UInt8(ord("9")):
                        break
                    acc = acc * 10 + UInt64(c) - UInt64(ord("0"))
                    i += 1
                var after = ptr[unsafe_offset=i] if i < n else UInt8(0)
                if not (
                    (after >= UInt8(ord("0")) and after <= UInt8(ord("9")))
                    or after == UInt8(ord("."))
                    or after == UInt8(ord("e"))
                    or after == UInt8(ord("E"))
                ):
                    self.pos = i
                    return _narrow[dtype](acc, negative)
        var parsed = parse_int_checked[dtype](self.data, self.pos, n)
        self.pos = parsed[1]
        return parsed[0]

    @always_inline
    def read_float[dtype: DType](mut self) raises -> Scalar[dtype]:
        self.skip_ws()
        var parsed = parse_float_span(self.data, self.pos, len(self.data))
        self.pos = parsed[1]
        return Scalar[dtype](parsed[0])

    # --- skipping ----------------------------------------------------

    def skip_value(mut self) raises -> Tuple[Int, Int]:
        """Consume one value without binding it. Returns its byte range.

        Used for a member no field wants, and to hand a subtree to a
        type that parses itself. Validating rather than merely counting
        brackets matters: an unknown field is still part of the
        document, and accepting rubbish inside it would mean two
        readers of the same bytes disagree about whether they are JSON.
        """
        self.skip_ws()
        var start = self.pos
        var c = self.peek()
        if c == UInt8(ord("{")):
            self.expect_object_begin()
            var first = True
            while self.next_member(first):
                first = False
                _ = self.read_key()
                _ = self.skip_value()
        elif c == UInt8(ord("[")):
            self.expect_array_begin()
            var first = True
            while self.next_element(first):
                first = False
                _ = self.skip_value()
        elif c == _QUOTE:
            _ = self._string_bounds()
        elif c == UInt8(ord("t")) or c == UInt8(ord("f")):
            _ = self.read_bool()
        elif c == UInt8(ord("n")):
            self.read_null()
        else:
            var token = scan_number(self.data, self.pos, len(self.data))
            if token.kind == NUM_INVALID:
                if token.err == NUM_ERR_LEADING_ZERO:
                    raise self.error_at("leading zeros in number", token.end)
                raise self.error_at("invalid number", token.end)
            self.pos = token.end
        return (start, self.pos)
