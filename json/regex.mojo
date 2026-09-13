# json - I-Regexp (RFC 9485) matching
#
# RFC 9535 JSONPath needs `match()` and `search()` function extensions,
# and the `pattern` keyword of JSON Schema needs the same dialect. Both
# are defined against I-Regexp, a deliberately small subset of XSD
# regular expressions with no backreferences, no lookaround and no lazy
# quantifiers. That subset is small enough to implement here rather
# than pull in a general regular expression library.
#
# The engine is a Thompson NFA. A pattern is parsed by recursive
# descent straight into instruction fragments, and matching simulates
# every alternative at once through a current and a next state set.
# There is no backtracking, so running time is bounded by
# `len(text) * len(program)` on every input including the adversarial
# ones (`(a+)+b` and friends) that a backtracking engine cannot
# survive. The price is that captures are not available, which is fine
# because neither caller needs them.
#
# Jump targets inside a fragment are stored relative to the
# instruction holding them. That is what makes fragments freely
# relocatable: building `a?` or `(ab){3}` is a matter of copying and
# concatenating fragments, with no offset patching pass.
#
# Matching is over Unicode scalar values, not bytes, as RFC 9485
# requires. The pattern is decoded to code points up front and the
# subject text is decoded as the simulation walks it.
#
# Deviations from RFC 9485 are noted at their definitions:
# `_single_char_escape` (accepts `\$`) and `_multi_char_ranges`
# (accepts `\d \D \s \S \w \W`, which the RFC drops, and with ASCII
# rather than Unicode-category meanings).

from std.collections import List


# Instruction opcodes. `CHAR`, `ANY`, `CLASS` and `NCLASS` consume one
# code point; `SPLIT` and `JUMP` are epsilon transitions resolved
# during the closure; `MATCH` reports success.
comptime _OP_CHAR: UInt8 = 0
comptime _OP_ANY: UInt8 = 1
comptime _OP_CLASS: UInt8 = 2
comptime _OP_NCLASS: UInt8 = 3
comptime _OP_SPLIT: UInt8 = 4
comptime _OP_JUMP: UInt8 = 5
comptime _OP_MATCH: UInt8 = 6

# A bounded quantifier is compiled by repeating the atom, so the bound
# is capped to keep a short pattern from producing a huge program.
comptime _MAX_QUANT = 1000

# Nested bounded quantifiers multiply, so the bound alone is not
# enough of a guard and the total program is capped as well.
comptime _MAX_PROGRAM = 100000

# Groups recurse, so nesting is capped to keep a hostile pattern off
# the parser's stack.
comptime _MAX_GROUP_DEPTH = 64

comptime _MAX_CODE_POINT = 0x10FFFF

# The replacement character stands in for malformed UTF-8 so that
# matching stays total on arbitrary input.
comptime _REPLACEMENT = 0xFFFD

comptime _CH_BACKSLASH = 0x5C
comptime _CH_LPAREN = 0x28
comptime _CH_RPAREN = 0x29
comptime _CH_STAR = 0x2A
comptime _CH_PLUS = 0x2B
comptime _CH_COMMA = 0x2C
comptime _CH_DASH = 0x2D
comptime _CH_DOT = 0x2E
comptime _CH_ZERO = 0x30
comptime _CH_NINE = 0x39
comptime _CH_QUESTION = 0x3F
comptime _CH_LBRACKET = 0x5B
comptime _CH_RBRACKET = 0x5D
comptime _CH_CARET = 0x5E
comptime _CH_LBRACE = 0x7B
comptime _CH_PIPE = 0x7C
comptime _CH_RBRACE = 0x7D


@fieldwise_init
struct _Inst(Copyable, ImplicitlyCopyable, Movable):
    """One NFA instruction.

    `a` and `b` carry the operand of a consuming instruction: a code
    point for `CHAR`, or a range-table slice for `CLASS` and `NCLASS`.
    `x` and `y` carry epsilon targets as offsets relative to this
    instruction's own index.
    """

    var op: UInt8
    var a: Int
    var b: Int
    var x: Int
    var y: Int


def _char_inst(code_point: Int) -> _Inst:
    return _Inst(_OP_CHAR, code_point, 0, 0, 0)


def _any_inst() -> _Inst:
    return _Inst(_OP_ANY, 0, 0, 0, 0)


def _class_inst(range_start: Int, range_count: Int, negated: Bool) -> _Inst:
    var op = _OP_NCLASS if negated else _OP_CLASS
    return _Inst(op, range_start, range_count, 0, 0)


def _split_inst(x: Int, y: Int) -> _Inst:
    return _Inst(_OP_SPLIT, 0, 0, x, y)


def _jump_inst(x: Int) -> _Inst:
    return _Inst(_OP_JUMP, 0, 0, x, 0)


def _match_inst() -> _Inst:
    return _Inst(_OP_MATCH, 0, 0, 0, 0)


def _single(var inst: _Inst) -> List[_Inst]:
    """Wrap one instruction as a fragment."""
    var frag = List[_Inst](capacity=1)
    frag.append(inst)
    return frag^


# UTF-8 decoding.
#
# json/unicode.mojo encodes code points but has no decoder, because
# every other path in the library works on bytes and never needs one.


def _is_continuation(byte: UInt8) -> Bool:
    return (byte & 0xC0) == 0x80


def _next_code_point(data: Span[UInt8, _], mut pos: Int) -> Int:
    """Decode the scalar value at `pos` and advance past it.

    Malformed sequences yield U+FFFD and advance a single byte. A
    matcher fed fuzzed input must terminate on every byte string, so
    rejecting the input is not an option here.
    """
    var n = len(data)
    var b0 = Int(data[pos])

    if b0 < 0x80:
        pos += 1
        return b0

    if b0 >= 0xC2 and b0 <= 0xDF:
        if pos + 1 < n and _is_continuation(data[pos + 1]):
            var cp = ((b0 & 0x1F) << 6) | (Int(data[pos + 1]) & 0x3F)
            pos += 2
            return cp
    elif b0 >= 0xE0 and b0 <= 0xEF:
        if (
            pos + 2 < n
            and _is_continuation(data[pos + 1])
            and _is_continuation(data[pos + 2])
        ):
            var cp = (
                ((b0 & 0x0F) << 12)
                | ((Int(data[pos + 1]) & 0x3F) << 6)
                | (Int(data[pos + 2]) & 0x3F)
            )
            # Overlong forms and surrogates are not scalar values.
            if cp >= 0x800 and not (cp >= 0xD800 and cp <= 0xDFFF):
                pos += 3
                return cp
    elif b0 >= 0xF0 and b0 <= 0xF4:
        if (
            pos + 3 < n
            and _is_continuation(data[pos + 1])
            and _is_continuation(data[pos + 2])
            and _is_continuation(data[pos + 3])
        ):
            var cp = (
                ((b0 & 0x07) << 18)
                | ((Int(data[pos + 1]) & 0x3F) << 12)
                | ((Int(data[pos + 2]) & 0x3F) << 6)
                | (Int(data[pos + 3]) & 0x3F)
            )
            if cp >= 0x10000 and cp <= _MAX_CODE_POINT:
                pos += 4
                return cp

    pos += 1
    return _REPLACEMENT


def _decode_all(text: String) -> List[Int]:
    """Decode a whole string to scalar values.

    Used for the pattern only. Parsing looks ahead by one or two
    characters in several places, and doing that over a byte stream
    would mean re-decoding constantly for no gain on a string that is
    read once at compile time.
    """
    var data = text.as_bytes()
    var n = len(data)
    var out = List[Int](capacity=n)
    var pos = 0
    while pos < n:
        out.append(_next_code_point(data, pos))
    return out^


def _describe(code_point: Int) -> String:
    """Render a code point for an error message."""
    if code_point >= 0x20 and code_point < 0x7F:
        return chr(code_point)
    return "U+" + hex(code_point).removeprefix("0x").upper()


# Fragment combinators.
#
# Each takes ownership of its operand and returns a fragment whose
# jumps are still relative, so the result can itself be an operand.


def _concat(var left: List[_Inst], var right: List[_Inst]) -> List[_Inst]:
    left.extend(right^)
    return left^


def _alternate(var left: List[_Inst], var right: List[_Inst]) -> List[_Inst]:
    """Build `left|right`.

    Layout is `SPLIT`, left, `JUMP` past right, right.
    """
    var left_len = len(left)
    var right_len = len(right)
    var out = List[_Inst](capacity=left_len + right_len + 2)
    out.append(_split_inst(1, left_len + 2))
    out.extend(left^)
    out.append(_jump_inst(right_len + 1))
    out.extend(right^)
    return out^


def _optional(var frag: List[_Inst]) -> List[_Inst]:
    """Build `frag?` as a `SPLIT` that may skip the fragment."""
    var n = len(frag)
    var out = List[_Inst](capacity=n + 1)
    out.append(_split_inst(1, n + 1))
    out.extend(frag^)
    return out^


def _star(var frag: List[_Inst]) -> List[_Inst]:
    """Build `frag*` as a `SPLIT` guarding a fragment that loops back."""
    var n = len(frag)
    var out = List[_Inst](capacity=n + 2)
    out.append(_split_inst(1, n + 2))
    out.extend(frag^)
    out.append(_jump_inst(-(n + 1)))
    return out^


def _plus(var frag: List[_Inst]) -> List[_Inst]:
    """Build `frag+` as the fragment followed by a looping `SPLIT`."""
    var n = len(frag)
    var out = List[_Inst](capacity=n + 1)
    out.extend(frag^)
    out.append(_split_inst(-n, 1))
    return out^


# Escape tables.


def _single_char_escape(code_point: Int) -> Int:
    """Resolve a single-character escape, or return -1.

    RFC 9485 lists `( ) * + - . ? [ \\ ] ^ n r t { | }`. `$` is
    accepted on top of that: it is a literal in I-Regexp either way, so
    writing it escaped is unambiguous and callers porting a pattern
    from a dialect where `$` is an anchor tend to write it that way.
    """
    if code_point == 0x6E:  # n
        return 0x0A
    if code_point == 0x72:  # r
        return 0x0D
    if code_point == 0x74:  # t
        return 0x09
    if code_point >= _CH_LPAREN and code_point <= _CH_PLUS:  # ( ) * +
        return code_point
    if code_point == _CH_DASH or code_point == _CH_DOT:
        return code_point
    if code_point == _CH_QUESTION:
        return code_point
    if code_point >= _CH_LBRACKET and code_point <= _CH_CARET:  # [ \ ] ^
        return code_point
    if code_point >= _CH_LBRACE and code_point <= _CH_RBRACE:  # { | }
        return code_point
    if code_point == 0x24:  # $
        return code_point
    return -1


def _multi_char_escape_kind(code_point: Int) -> Int:
    """Classify `\\d \\D \\s \\S \\w \\W`, or return 0.

    The sign carries the complement, so `\\D` is the negation of `\\d`.
    """
    if code_point == 0x64:  # d
        return 1
    if code_point == 0x44:  # D
        return -1
    if code_point == 0x73:  # s
        return 2
    if code_point == 0x53:  # S
        return -2
    if code_point == 0x77:  # w
        return 3
    if code_point == 0x57:  # W
        return -3
    return 0


def _complement_ranges(var src: List[Int]) -> List[Int]:
    """Invert a sorted, disjoint range list over the scalar range."""
    var out = List[Int](capacity=len(src) + 2)
    var next_lo = 0
    var k = 0
    while k < len(src):
        var lo = src[k]
        if lo > next_lo:
            out.append(next_lo)
            out.append(lo - 1)
        next_lo = src[k + 1] + 1
        k += 2
    if next_lo <= _MAX_CODE_POINT:
        out.append(next_lo)
        out.append(_MAX_CODE_POINT)
    return out^


def _multi_char_ranges(kind: Int) -> List[Int]:
    """Range pairs for a multi-character escape.

    RFC 9485 drops these escapes entirely in favour of `\\p{...}`
    category classes, but every JSON Schema `pattern` in the wild uses
    them and no caller of this library has asked for category classes.
    They are defined over ASCII here. The XSD definitions are in terms
    of Unicode general categories, which would need category tables
    measured in megabytes to answer a question no caller is asking.
    Note that this makes `\\w` include `_`, which the XSD reading
    excludes because `_` is connector punctuation.
    """
    var base = List[Int]()
    var abs_kind = kind if kind > 0 else -kind

    if abs_kind == 1:
        base.append(_CH_ZERO)
        base.append(_CH_NINE)
    elif abs_kind == 2:
        # XSD `\s` is exactly tab, line feed, carriage return and space.
        base.append(0x09)
        base.append(0x0A)
        base.append(0x0D)
        base.append(0x0D)
        base.append(0x20)
        base.append(0x20)
    else:
        base.append(_CH_ZERO)
        base.append(_CH_NINE)
        base.append(0x41)  # A
        base.append(0x5A)  # Z
        base.append(0x5F)  # _
        base.append(0x5F)
        base.append(0x61)  # a
        base.append(0x7A)  # z

    if kind < 0:
        return _complement_ranges(base^)
    return base^


struct _Parser(Movable):
    """Recursive descent over the pattern, emitting fragments.

    The parser owns the shared range table because character classes
    from anywhere in the pattern append to it and instructions refer
    into it by slice.
    """

    var pat: List[Int]
    var pos: Int
    var ranges: List[Int]
    var depth: Int

    def __init__(out self, pattern: String):
        self.pat = _decode_all(pattern)
        self.pos = 0
        self.ranges = List[Int]()
        self.depth = 0

    def _peek(self) -> Int:
        """Current code point, or -1 at the end of the pattern."""
        if self.pos >= len(self.pat):
            return -1
        return self.pat[self.pos]

    def _peek_ahead(self, offset: Int) -> Int:
        if self.pos + offset >= len(self.pat):
            return -1
        return self.pat[self.pos + offset]

    def parse(mut self) raises -> List[_Inst]:
        var frag = self._alternation()
        if self.pos < len(self.pat):
            # `_branch` stops at `)` and at `|`, and `|` is always
            # consumed, so anything left over is a stray `)`.
            raise Error("unbalanced ')' in pattern")
        frag.append(_match_inst())
        return frag^

    def _alternation(mut self) raises -> List[_Inst]:
        var frag = self._branch()
        while self._peek() == _CH_PIPE:
            self.pos += 1
            var right = self._branch()
            frag = _alternate(frag^, right^)
            if len(frag) > _MAX_PROGRAM:
                raise Error("pattern compiles to too many instructions")
        return frag^

    def _branch(mut self) raises -> List[_Inst]:
        var frag = List[_Inst]()
        while True:
            var c = self._peek()
            if c < 0 or c == _CH_PIPE or c == _CH_RPAREN:
                break
            var piece = self._piece()
            frag = _concat(frag^, piece^)
            if len(frag) > _MAX_PROGRAM:
                raise Error("pattern compiles to too many instructions")
        return frag^

    def _piece(mut self) raises -> List[_Inst]:
        var atom = self._atom()
        var c = self._peek()

        if c == _CH_QUESTION:
            self.pos += 1
            self._reject_second_quantifier()
            return _optional(atom^)
        if c == _CH_STAR:
            self.pos += 1
            self._reject_second_quantifier()
            return _star(atom^)
        if c == _CH_PLUS:
            self.pos += 1
            self._reject_second_quantifier()
            return _plus(atom^)
        if c == _CH_LBRACE:
            return self._range_quantifier(atom^)
        return atom^

    def _reject_second_quantifier(self) raises:
        """A piece carries at most one quantifier.

        This is what rules out the lazy and possessive forms `a*?` and
        `a*+`, neither of which I-Regexp has.
        """
        var c = self._peek()
        if (
            c == _CH_QUESTION
            or c == _CH_STAR
            or c == _CH_PLUS
            or c == _CH_LBRACE
        ):
            raise Error(
                "'"
                + _describe(c)
                + "' follows another quantifier; a piece takes only one"
            )

    def _atom(mut self) raises -> List[_Inst]:
        var c = self._peek()
        if c < 0:
            raise Error("pattern ends where an atom was expected")

        if c == _CH_LPAREN:
            self.pos += 1
            self.depth += 1
            if self.depth > _MAX_GROUP_DEPTH:
                raise Error("pattern nests groups more than 64 deep")
            var inner = self._alternation()
            if self._peek() != _CH_RPAREN:
                raise Error("unbalanced '(' in pattern")
            self.pos += 1
            self.depth -= 1
            return inner^

        if c == _CH_DOT:
            self.pos += 1
            return _single(_any_inst())

        if c == _CH_LBRACKET:
            self.pos += 1
            return self._char_class()

        if c == _CH_BACKSLASH:
            return self._escape_atom()

        if c == _CH_STAR or c == _CH_PLUS or c == _CH_QUESTION:
            raise Error(
                "'" + _describe(c) + "' has no atom to quantify",
            )
        if c == _CH_LBRACE or c == _CH_RBRACE:
            raise Error(
                "'"
                + _describe(c)
                + "' is a metacharacter; write '\\"
                + _describe(c)
                + "' to match it literally"
            )
        if c == _CH_RBRACKET:
            raise Error(
                "']' is a metacharacter; write '\\]' to match it literally"
            )

        self.pos += 1
        return _single(_char_inst(c))

    def _escape_atom(mut self) raises -> List[_Inst]:
        var e = self._peek_ahead(1)
        if e < 0:
            raise Error("pattern ends with a trailing backslash")

        var literal = _single_char_escape(e)
        if literal >= 0:
            self.pos += 2
            return _single(_char_inst(literal))

        var kind = _multi_char_escape_kind(e)
        if kind == 0:
            raise Error("unknown escape '\\" + _describe(e) + "' in pattern")

        self.pos += 2
        var start = len(self.ranges) // 2
        var count = self._append_ranges(_multi_char_ranges(kind))
        return _single(_class_inst(start, count, False))

    def _append_ranges(mut self, var src: List[Int]) -> Int:
        """Append range pairs to the shared table, returning the count."""
        var pairs = len(src) // 2
        for k in range(len(src)):
            self.ranges.append(src[k])
        return pairs

    def _class_member(mut self) raises -> Int:
        """Consume one class member that can stand as a range endpoint.

        Returns the code point, or -1 when the member was a
        multi-character escape, whose ranges have already been added.
        """
        var c = self._peek()
        if c != _CH_BACKSLASH:
            self.pos += 1
            return c

        var e = self._peek_ahead(1)
        if e < 0:
            raise Error("pattern ends with a trailing backslash")

        var literal = _single_char_escape(e)
        if literal >= 0:
            self.pos += 2
            return literal

        var kind = _multi_char_escape_kind(e)
        if kind == 0:
            raise Error(
                "unknown escape '\\" + _describe(e) + "' in character class"
            )

        self.pos += 2
        _ = self._append_ranges(_multi_char_ranges(kind))
        return -1

    def _char_class(mut self) raises -> List[_Inst]:
        """Parse the body of `[...]`, the bracket already consumed."""
        var negated = False
        if self._peek() == _CH_CARET:
            negated = True
            self.pos += 1

        var start = len(self.ranges) // 2
        var count = 0
        var n = len(self.pat)

        while True:
            if self.pos >= n:
                raise Error("unterminated character class: missing ']'")
            if self.pat[self.pos] == _CH_RBRACKET:
                self.pos += 1
                break

            var before = len(self.ranges) // 2
            var lo = self._class_member()
            if lo < 0:
                # A multi-character escape contributes its own ranges
                # and cannot be a range endpoint.
                count += len(self.ranges) // 2 - before
                continue

            # A `-` is a range only with a member after it. Trailing and
            # leading dashes are literals, which the ABNF allows.
            if (
                self._peek() == _CH_DASH
                and self._peek_ahead(1) != _CH_RBRACKET
                and self._peek_ahead(1) >= 0
            ):
                self.pos += 1
                var hi = self._class_member()
                if hi < 0:
                    raise Error(
                        "a multi-character escape cannot end a character"
                        " class range"
                    )
                if hi < lo:
                    raise Error(
                        "character class range '"
                        + _describe(lo)
                        + "-"
                        + _describe(hi)
                        + "' runs backwards"
                    )
                self.ranges.append(lo)
                self.ranges.append(hi)
            else:
                self.ranges.append(lo)
                self.ranges.append(lo)
            count += 1

        if count == 0:
            raise Error("empty character class")

        return _single(_class_inst(start, count, negated))

    def _quant_exact(mut self) raises -> Int:
        """Read the decimal count inside `{...}`.

        The accumulator is clamped so a long run of digits reports the
        cap rather than wrapping around.
        """
        var n = len(self.pat)
        var start = self.pos
        var value = 0
        while self.pos < n:
            var c = self.pat[self.pos]
            if c < _CH_ZERO or c > _CH_NINE:
                break
            if value <= _MAX_QUANT:
                value = value * 10 + (c - _CH_ZERO)
            self.pos += 1
        if self.pos == start:
            raise Error("'{' quantifier needs a decimal count")
        return value

    def _range_quantifier(
        mut self, var atom: List[_Inst]
    ) raises -> List[_Inst]:
        """Parse `{n}`, `{n,}` or `{n,m}` and expand the atom."""
        self.pos += 1  # past '{'
        var lo = self._quant_exact()
        var hi = lo

        if self._peek() == _CH_COMMA:
            self.pos += 1
            if self._peek() == _CH_RBRACE:
                hi = -1  # unbounded
            else:
                hi = self._quant_exact()

        if self._peek() != _CH_RBRACE:
            raise Error("unbalanced '{' in pattern: missing '}'")
        self.pos += 1
        self._reject_second_quantifier()

        if lo > _MAX_QUANT or hi > _MAX_QUANT:
            raise Error(
                "quantifier bound exceeds the limit of "
                + String(_MAX_QUANT)
                + "; the atom is repeated to compile it"
            )
        if hi >= 0 and hi < lo:
            raise Error(
                "quantifier range runs backwards: the low bound"
                " exceeds the high bound"
            )

        # Refuse before building rather than after, so that nested
        # bounded quantifiers cannot allocate their way to the cap.
        var unit = len(atom)
        var copies = hi if hi >= 0 else lo
        if unit * (copies + 1) + copies + 2 > _MAX_PROGRAM:
            raise Error("pattern compiles to too many instructions")

        if hi < 0:
            if lo == 0:
                return _star(atom^)
            # `_plus` supplies the last mandatory repetition along with
            # the loop, so only `lo - 1` copies precede it.
            var head = List[_Inst]()
            for _ in range(lo - 1):
                head.extend(atom.copy())
            return _concat(head^, _plus(atom^))

        var out = List[_Inst]()
        for _ in range(lo):
            out.extend(atom.copy())
        # The optional tail is a flat run rather than a nest, because
        # without captures `F?F?` accepts exactly the same strings as
        # `(F(F)?)?` and is cheaper to build.
        for _ in range(hi - lo):
            out = _concat(out^, _optional(atom.copy()))
        return out^


struct Regex(Movable):
    """A compiled I-Regexp pattern.

    Compile once and match many times: nothing in `full_match` or
    `search` mutates the program, and both allocate only the small
    per-run state sets.
    """

    var prog: List[_Inst]
    var ranges: List[Int]

    def __init__(out self, var prog: List[_Inst], var ranges: List[Int]):
        self.prog = prog^
        self.ranges = ranges^

    @staticmethod
    def compile(pattern: String) raises -> Regex:
        """Compile an I-Regexp pattern.

        Args:
            pattern: The pattern, in RFC 9485 syntax.

        Returns:
            The compiled matcher.

        Raises:
            If the pattern is not a valid I-Regexp, or if it would
            expand past the instruction cap.
        """
        var parser = _Parser(pattern)
        var prog = parser.parse()
        return Regex(prog^, parser.ranges.copy())

    def full_match(self, text: String) -> Bool:
        """Whether the whole of `text` matches.

        This is I-Regexp's own semantics: a pattern is anchored to both
        ends of the subject, which is what RFC 9535's `match()` needs.
        """
        return self._run(text, search_mode=False)

    def search(self, text: String) -> Bool:
        """Whether any substring of `text` matches.

        This is what RFC 9535's `search()` needs. It is built here
        rather than by wrapping the pattern in `.*`, because starting a
        thread at every position costs one epsilon closure per
        character instead of a second pass over the program.
        """
        return self._run(text, search_mode=True)

    def _accepts(self, inst: _Inst, code_point: Int) -> Bool:
        if inst.op == _OP_CHAR:
            return code_point == inst.a
        if inst.op == _OP_ANY:
            # RFC 9485 section 3 maps an unescaped dot to `[^\n\r]`.
            return code_point != 0x0A and code_point != 0x0D
        if inst.op == _OP_CLASS or inst.op == _OP_NCLASS:
            var found = False
            for k in range(inst.b):
                var lo = self.ranges[(inst.a + k) * 2]
                var hi = self.ranges[(inst.a + k) * 2 + 1]
                if code_point >= lo and code_point <= hi:
                    found = True
                    break
            if inst.op == _OP_CLASS:
                return found
            return not found
        return False

    def _add_thread(
        self,
        mut threads: List[Int],
        mut marks: List[Int],
        mut stack: List[Int],
        generation: Int,
        entry: Int,
    ):
        """Add the epsilon closure of `entry` to `threads`.

        `marks` records the generation each instruction was last added
        in. That both keeps the state set free of duplicates and
        terminates the closure over a pattern like `()*` whose epsilon
        transitions form a cycle.
        """
        stack.clear()
        stack.append(entry)
        while len(stack) > 0:
            var pc = stack.pop()
            if marks[pc] == generation:
                continue
            marks[pc] = generation
            var inst = self.prog[pc]
            if inst.op == _OP_JUMP:
                stack.append(pc + inst.x)
            elif inst.op == _OP_SPLIT:
                stack.append(pc + inst.y)
                stack.append(pc + inst.x)
            else:
                threads.append(pc)

    def _run(self, text: String, search_mode: Bool) -> Bool:
        var data = text.as_bytes()
        var n = len(data)
        var size = len(self.prog)

        var marks = List[Int](length=size, fill=-1)
        var current = List[Int](capacity=size)
        var next = List[Int](capacity=size)
        var stack = List[Int](capacity=size)

        var generation = 0
        var pos = 0
        self._add_thread(current, marks, stack, generation, 0)

        while True:
            for k in range(len(current)):
                if self.prog[current[k]].op == _OP_MATCH:
                    # Under `full_match` a match only counts once the
                    # whole subject has been consumed.
                    if search_mode or pos == n:
                        return True
            if pos == n:
                return False
            if not search_mode and len(current) == 0:
                return False

            var code_point = _next_code_point(data, pos)
            generation += 1
            next.clear()
            for k in range(len(current)):
                var pc = current[k]
                if self._accepts(self.prog[pc], code_point):
                    self._add_thread(next, marks, stack, generation, pc + 1)
            swap(current, next)

            if search_mode:
                # Seeding a thread at every position is what turns the
                # anchored program into a substring search, and the
                # generation marks stop it from duplicating threads
                # that are already live.
                self._add_thread(current, marks, stack, generation, 0)


def regex_full_match(pattern: String, text: String) raises -> Bool:
    """Compile `pattern` and test it against the whole of `text`.

    Args:
        pattern: An I-Regexp pattern.
        text: The subject string.

    Returns:
        True when the whole subject matches.

    Raises:
        If the pattern is not a valid I-Regexp.
    """
    return Regex.compile(pattern).full_match(text)


def regex_search(pattern: String, text: String) raises -> Bool:
    """Compile `pattern` and test it against every substring of `text`.

    Args:
        pattern: An I-Regexp pattern.
        text: The subject string.

    Returns:
        True when any substring matches.

    Raises:
        If the pattern is not a valid I-Regexp.
    """
    return Regex.compile(pattern).search(text)
