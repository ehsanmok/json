# json - JSONPath (RFC 9535, March 2024)
#
# The previous implementation was a single pass that turned a path into
# a flat token list and applied the tokens one at a time. That shape
# cannot express the grammar RFC 9535 standardised. A bracketed segment
# is a comma-separated list of selectors, so one bracket pair can carry
# several tokens; a filter holds a logical expression tree with its own
# nested queries; and a comparison operand must be a singular query,
# which is a property of a whole sub-query rather than of any one
# token. Trying to keep the flat shape is what produced the family of
# defects that motivated this rewrite: unions dropped on the floor,
# slice bounds confusing "unspecified" with "negative", a negative
# index resolved once and reused across every later array, a filter
# scanner that counted brackets without noticing it was inside a string
# literal, and a catch-all that advanced past any byte it did not
# recognise so that invalid queries parsed as valid ones.
#
# So this is a parser and an evaluator, in that order. The parser
# builds an abstract syntax tree and rejects everything the ABNF in
# RFC 9535 section 2.1 rejects, including the well-typedness rules for
# function extensions in section 2.4.3. The evaluator then walks the
# tree, and every case it can reach is one the parser has already
# proved is meaningful.
#
# The tree is stored as parallel `List`s of nodes addressed by integer
# index rather than as a graph of pointers. A selector may hold a
# filter, a filter holds an expression, an expression holds queries,
# and those queries hold selectors again, so the type is mutually
# recursive; Mojo needs indirection for that, and flat pools are the
# cheapest indirection available. They also make `JSONPath` a plain
# movable value that can be compiled once and evaluated many times.
#
# Evaluation carries a normalized path (section 2.7) alongside every
# value, which is what lets `paths` and `query_with_paths` exist at no
# extra traversal. Note that a normalized path is not a JSON Pointer:
# it is `$['a'][0]` with single quotes, and `json/pointer.mojo` is a
# different syntax for a different specification.
#
# Known deviations, all deliberate:
#
#   - Section 2.3.5.1 spells a singular query's segments without
#     interior whitespace, so `$[?@['a' ] == 1]` is strictly invalid.
#     This accepts it. Rejecting it would mean threading a "saw a
#     space" flag out of the bracket parser purely to reject a query
#     that has an unambiguous meaning.
#   - `match()` and `search()` treat a pattern that is not a valid
#     I-Regexp as producing LogicalFalse rather than as an error, at
#     parse time as well as at run time. RFC 9535 section 2.4.6 defines
#     the result only for valid patterns, and false is the answer that
#     keeps a filter total.
#   - A member name shorthand accepts any byte at or above 0x80 as a
#     name character. The ABNF says `%x80-10FFFF`, which is the same
#     set for well-formed UTF-8 input and a superset for input that is
#     not well-formed.

from std.collections import List

from .regex import Regex
from .value import Null, Value


# Byte values the parser tests against. `_Scan.peek` reports -1 past
# the end of the query so that every comparison here is safe without a
# separate bounds test at the call site.
comptime _C_TAB = 0x09
comptime _C_LF = 0x0A
comptime _C_CR = 0x0D
comptime _C_SPACE = 0x20
comptime _C_BANG = 0x21
comptime _C_DQUOTE = 0x22
comptime _C_DOLLAR = 0x24
comptime _C_AMP = 0x26
comptime _C_SQUOTE = 0x27
comptime _C_LPAREN = 0x28
comptime _C_RPAREN = 0x29
comptime _C_STAR = 0x2A
comptime _C_PLUS = 0x2B
comptime _C_COMMA = 0x2C
comptime _C_MINUS = 0x2D
comptime _C_DOT = 0x2E
comptime _C_SLASH = 0x2F
comptime _C_0 = 0x30
comptime _C_9 = 0x39
comptime _C_COLON = 0x3A
comptime _C_LT = 0x3C
comptime _C_EQ = 0x3D
comptime _C_GT = 0x3E
comptime _C_QUESTION = 0x3F
comptime _C_AT = 0x40
comptime _C_UPPER_A = 0x41
comptime _C_UPPER_Z = 0x5A
comptime _C_LBRACKET = 0x5B
comptime _C_BACKSLASH = 0x5C
comptime _C_RBRACKET = 0x5D
comptime _C_UNDERSCORE = 0x5F
comptime _C_LOWER_A = 0x61
comptime _C_LOWER_B = 0x62
comptime _C_LOWER_E = 0x65
comptime _C_LOWER_F = 0x66
comptime _C_LOWER_N = 0x6E
comptime _C_LOWER_R = 0x72
comptime _C_LOWER_T = 0x74
comptime _C_LOWER_U = 0x75
comptime _C_LOWER_Z = 0x7A
comptime _C_PIPE = 0x7C

# Selector kinds.
comptime _SEL_NAME = 0
comptime _SEL_WILDCARD = 1
comptime _SEL_INDEX = 2
comptime _SEL_SLICE = 3
comptime _SEL_FILTER = 4

# Logical expression node kinds.
comptime _EX_OR = 0
comptime _EX_AND = 1
comptime _EX_NOT = 2
comptime _EX_TEST = 3
comptime _EX_FUNC_TEST = 4
comptime _EX_COMPARE = 5

# Comparison operators, in the order RFC 9535 section 2.3.5.1 lists
# them.
comptime _OP_EQ = 0
comptime _OP_NE = 1
comptime _OP_LE = 2
comptime _OP_GE = 3
comptime _OP_LT = 4
comptime _OP_GT = 5

# Operand kinds, shared by comparison operands and function arguments
# so that an operand can be stored in either pool without translation.
comptime _OPD_LITERAL = 0
comptime _OPD_QUERY = 1
comptime _OPD_FUNC = 2
comptime _OPD_LOGICAL = 3

# The declared types of RFC 9535 section 2.4.1.
comptime _TY_VALUE = 0
comptime _TY_LOGICAL = 1
comptime _TY_NODES = 2

# The five function extensions of section 2.4.
comptime _FN_LENGTH = 0
comptime _FN_COUNT = 1
comptime _FN_MATCH = 2
comptime _FN_SEARCH = 3
comptime _FN_VALUE = 4

# Section 2.1 bounds an index or a slice bound by the I-JSON safe
# integer range, so a query naming a position no array could ever have
# is a parse error rather than a silent miss.
comptime _MAX_SAFE_INT = 9007199254740991

# A bracketed selection may contain a filter, which may contain a query
# with another bracketed selection. This bounds that nesting so a
# pathological query cannot exhaust the parser's own stack.
comptime _MAX_QUERY_DEPTH = 64

# The matching bound during evaluation. Filter nesting is already
# capped at parse time, so this only ever fires if the two caps drift
# apart.
comptime _MAX_EVAL_DEPTH = 256

# A descendant segment walks the document, which is parsed without a
# nesting limit by default. The walk below is iterative, so depth costs
# heap rather than stack, but an unbounded walk is still not something
# a query should be able to ask for.
comptime _MAX_DESCENT_DEPTH = 10000

# Deep equality recurses through both operands together, so it needs
# its own bound.
comptime _MAX_EQUAL_DEPTH = 256


struct _Selector(Copyable):
    """One selector from RFC 9535 section 2.3.

    The fields are shared across the five kinds rather than split into
    a variant per kind, because Mojo has no sum type and five parallel
    pools would cost more than four unused integers.
    """

    var kind: Int
    var name: String
    # Index for `_SEL_INDEX`, slice start for `_SEL_SLICE`, and the
    # root expression node for `_SEL_FILTER`.
    var a: Int
    var b: Int
    var c: Int
    # Whether the slice wrote a start and an end. Keeping these
    # separate from the values is the whole point: a missing bound and
    # a bound of -1 mean different things, and conflating them is what
    # made every open-ended slice drop its last element.
    var has_a: Bool
    var has_b: Bool

    def __init__(
        out self,
        kind: Int,
        name: String = String(),
        a: Int = 0,
        b: Int = 0,
        c: Int = 1,
        has_a: Bool = False,
        has_b: Bool = False,
    ):
        self.kind = kind
        self.name = name.copy()
        self.a = a
        self.b = b
        self.c = c
        self.has_a = has_a
        self.has_b = has_b


@fieldwise_init
struct _Segment(Copyable):
    """A child or descendant segment, naming a run of selectors."""

    var descendant: Bool
    var sel_start: Int
    var sel_count: Int


@fieldwise_init
struct _QueryRef(Copyable):
    """A query: a run of segments applied from `$` or from `@`.

    `singular` records the section 2.3.5.1 property, decided when the
    query is parsed because that is the only place the shape of every
    segment is known at once.
    """

    var seg_start: Int
    var seg_count: Int
    var from_root: Bool
    var singular: Bool


@fieldwise_init
struct _Expr(Copyable):
    """A node of a filter's logical expression tree."""

    var kind: Int
    var a: Int
    var b: Int
    var op: Int


@fieldwise_init
struct _Operand(Copyable):
    """A parsed comparable or function argument, before it is stored.

    `singular` and `result` carry what the well-typedness checks need:
    whether a query operand is a singular query, and what type a
    function operand declares.
    """

    var kind: Int
    var index: Int
    var singular: Bool
    var result: Int


@fieldwise_init
struct _Slot(Copyable):
    """An operand as stored in a pool: a kind and an index into it."""

    var kind: Int
    var index: Int


@fieldwise_init
struct _Func(Copyable):
    """A call to one of the function extensions.

    `regex` caches the compiled pattern of `match` and `search` when
    the pattern is a string literal, which is the common case and the
    only one that can be compiled before a document is in hand. It is
    -1 when the pattern is not a literal and -2 when it is a literal
    that is not a valid I-Regexp.
    """

    var id: Int
    var arg_start: Int
    var arg_count: Int
    var regex: Int


@fieldwise_init
struct _Node(Copyable):
    """A node of a nodelist: a value and its normalized path."""

    var path: String
    var value: Value


@fieldwise_init
struct _Maybe(Copyable):
    """A value or the "Nothing" of RFC 9535 section 2.4.1.

    Nothing is what a singular query yields when the member or element
    it names is absent, and it compares equal only to itself. That is
    why it cannot be modelled as `null`, and why `@.missing != 1` is
    true rather than false.
    """

    var present: Bool
    var value: Value


def _nothing() -> _Maybe:
    return _Maybe(False, Value(Null()))


struct _Scan(Movable):
    """The cursor over the query text.

    Held apart from the tree being built so that the parsing methods
    can take the tree as `mut self` and the cursor as a second mutable
    argument, which is what lets them recurse.
    """

    var buf: List[UInt8]
    var text: String
    var pos: Int
    var depth: Int

    def __init__(out self, path: String):
        self.buf = List[UInt8]()
        var bytes = path.as_bytes()
        for i in range(len(bytes)):
            self.buf.append(bytes[i])
        self.text = path.copy()
        self.pos = 0
        self.depth = 0

    def peek(self) -> Int:
        """The byte under the cursor, or -1 at the end of the query."""
        if self.pos >= len(self.buf):
            return -1
        return Int(self.buf[self.pos])

    def peek_at(self, offset: Int) -> Int:
        if self.pos + offset >= len(self.buf):
            return -1
        return Int(self.buf[self.pos + offset])

    def at_end(self) -> Bool:
        return self.pos >= len(self.buf)

    def skip_ws(mut self):
        """Consume `S` from the ABNF: space, tab, line feed, return."""
        while True:
            var c = self.peek()
            if c == _C_SPACE or c == _C_TAB or c == _C_LF or c == _C_CR:
                self.pos += 1
            else:
                return

    def slice(self, start: Int, end: Int) -> String:
        var scratch = List[UInt8]()
        for i in range(start, end):
            scratch.append(self.buf[i])
        return String(unsafe_from_utf8=Span(scratch))

    def error(self, message: String) -> Error:
        return Error(
            "JSONPath: "
            + message
            + " at offset "
            + String(self.pos)
            + " in "
            + self.text
        )

    def word(mut self, literal: String) -> Bool:
        """Consume `literal` if it is next and not glued to a name.

        `true` and `truer` have to be told apart, and the second is not
        a keyword followed by rubbish; it is a token the caller should
        try to read some other way.
        """
        var bytes = literal.as_bytes()
        var n = len(bytes)
        if self.pos + n > len(self.buf):
            return False
        for i in range(n):
            if self.buf[self.pos + i] != bytes[i]:
                return False
        if _is_func_name_char(self.peek_at(n)):
            return False
        self.pos += n
        return True

    def comparison_op(mut self) -> Int:
        """Consume a comparison operator, or report -1.

        The two-byte operators are tested first so that `<=` is never
        read as `<` followed by a stray `=`.
        """
        var c = self.peek()
        var d = self.peek_at(1)
        if c == _C_EQ and d == _C_EQ:
            self.pos += 2
            return _OP_EQ
        if c == _C_BANG and d == _C_EQ:
            self.pos += 2
            return _OP_NE
        if c == _C_LT and d == _C_EQ:
            self.pos += 2
            return _OP_LE
        if c == _C_GT and d == _C_EQ:
            self.pos += 2
            return _OP_GE
        if c == _C_LT:
            self.pos += 1
            return _OP_LT
        if c == _C_GT:
            self.pos += 1
            return _OP_GT
        return -1

    def member_name(mut self) raises -> String:
        """Read a member name shorthand, per `member-name-shorthand`.

        The ABNF admits letters, `_` and non-ASCII as the first
        character and adds digits after it, which is what makes `$.1a`
        and `$.foo bar` invalid queries rather than odd member names.
        """
        if not _is_name_first(self.peek()):
            raise self.error("expected a member name")
        var start = self.pos
        self.pos += 1
        while _is_name_char(self.peek()):
            self.pos += 1
        return self.slice(start, self.pos)

    def parse_int(mut self) raises -> Int:
        """Read `int` from section 2.3.3.1: `0` or an unpadded integer."""
        var negative = False
        if self.peek() == _C_MINUS:
            negative = True
            self.pos += 1
        var c = self.peek()
        if not _is_digit(c):
            raise self.error("expected a digit")
        if c == _C_0:
            self.pos += 1
            if negative:
                raise self.error("'-0' is not a valid index")
            if _is_digit(self.peek()):
                raise self.error("an integer may not have a leading zero")
            return 0
        var value = 0
        while _is_digit(self.peek()):
            value = value * 10 + (self.peek() - _C_0)
            if value > _MAX_SAFE_INT:
                raise self.error("integer is outside the I-JSON safe range")
            self.pos += 1
        if negative:
            return -value
        return value

    def parse_number(mut self) raises -> Value:
        """Read `number` from section 2.3.5.1.

        Wider than `int`: `-0` is a number though it is not an index, a
        fraction and an exponent are allowed, and the exponent marker
        is lowercase `e` only.
        """
        var begin = self.pos
        if self.peek() == _C_MINUS:
            self.pos += 1
        var c = self.peek()
        if not _is_digit(c):
            raise self.error("expected a number")
        if c == _C_0:
            self.pos += 1
            if _is_digit(self.peek()):
                raise self.error("a number may not have a leading zero")
        else:
            while _is_digit(self.peek()):
                self.pos += 1
        var fractional = False
        if self.peek() == _C_DOT:
            fractional = True
            self.pos += 1
            if not _is_digit(self.peek()):
                raise self.error("a fraction needs at least one digit")
            while _is_digit(self.peek()):
                self.pos += 1
        if self.peek() == _C_LOWER_E:
            fractional = True
            self.pos += 1
            if self.peek() == _C_MINUS or self.peek() == _C_PLUS:
                self.pos += 1
            if not _is_digit(self.peek()):
                raise self.error("an exponent needs at least one digit")
            while _is_digit(self.peek()):
                self.pos += 1
        var text = self.slice(begin, self.pos)
        # An integer too wide for `Int` still has a value under IEEE
        # 754, and section 2.3.5.2.2 compares numbers by value, so it
        # is better read as a double than rejected.
        if fractional or text.byte_length() > 18:
            return Value(atof(text))
        return Value(atol(text))

    def parse_string(mut self) raises -> String:
        """Read `string-literal` from section 2.3.1.1.

        Both quote flavours take the whole JSON escape set. The quote
        that did not open the literal stands for itself and may not be
        escaped, which is the one asymmetry in the rule.
        """
        var quote = self.peek()
        self.pos += 1
        var out = List[UInt8]()
        while True:
            if self.at_end():
                raise self.error("unterminated string literal")
            var c = self.peek()
            if c == quote:
                self.pos += 1
                return String(unsafe_from_utf8=Span(out))
            if c < 0x20:
                raise self.error(
                    "a control character must be escaped in a string literal"
                )
            if c != _C_BACKSLASH:
                out.append(UInt8(c))
                self.pos += 1
                continue
            self.pos += 1
            var e = self.peek()
            if e == quote or e == _C_BACKSLASH or e == _C_SLASH:
                out.append(UInt8(e))
                self.pos += 1
            elif e == _C_LOWER_B:
                out.append(0x08)
                self.pos += 1
            elif e == _C_LOWER_F:
                out.append(0x0C)
                self.pos += 1
            elif e == _C_LOWER_N:
                out.append(0x0A)
                self.pos += 1
            elif e == _C_LOWER_R:
                out.append(0x0D)
                self.pos += 1
            elif e == _C_LOWER_T:
                out.append(0x09)
                self.pos += 1
            elif e == _C_LOWER_U:
                self.pos += 1
                var code = self.parse_hex4()
                if code >= 0xD800 and code <= 0xDBFF:
                    # A high surrogate is only meaningful as half of a
                    # pair, so the low half is required rather than
                    # optional.
                    if self.peek() != _C_BACKSLASH or self.peek_at(1) != (
                        _C_LOWER_U
                    ):
                        raise self.error(
                            "a high surrogate must be followed by a low"
                            " surrogate"
                        )
                    self.pos += 2
                    var low = self.parse_hex4()
                    if low < 0xDC00 or low > 0xDFFF:
                        raise self.error(
                            "a high surrogate must be followed by a low"
                            " surrogate"
                        )
                    code = 0x10000 + (code - 0xD800) * 0x400 + (low - 0xDC00)
                elif code >= 0xDC00 and code <= 0xDFFF:
                    raise self.error("a low surrogate may not stand alone")
                _encode_utf8(code, out)
            else:
                raise self.error("unknown escape in a string literal")

    def parse_hex4(mut self) raises -> Int:
        var value = 0
        for _ in range(4):
            var c = self.peek()
            var digit = _hex_value(c)
            if digit < 0:
                raise self.error("'\\u' needs four hexadecimal digits")
            value = value * 16 + digit
            self.pos += 1
        return value


struct JSONPath(Movable):
    """A compiled RFC 9535 query.

    Compiling separates the work that depends only on the query from
    the work that depends on the document, so a query used against many
    documents pays for parsing, validation and regular expression
    compilation once.
    """

    var segments: List[_Segment]
    var selectors: List[_Selector]
    var queries: List[_QueryRef]
    var exprs: List[_Expr]
    var cmps: List[_Slot]
    var funcs: List[_Func]
    var args: List[_Slot]
    var literals: List[Value]
    var regexes: List[Regex]
    var root: Int

    def __init__(out self):
        self.segments = List[_Segment]()
        self.selectors = List[_Selector]()
        self.queries = List[_QueryRef]()
        self.exprs = List[_Expr]()
        self.cmps = List[_Slot]()
        self.funcs = List[_Func]()
        self.args = List[_Slot]()
        self.literals = List[Value]()
        self.regexes = List[Regex]()
        self.root = -1

    @staticmethod
    def compile(path: String) raises -> JSONPath:
        """Parse and validate a query.

        Args:
            path: The query, which must be a complete
                `jsonpath-query` with no leading or trailing
                whitespace.

        Returns:
            The compiled query.

        Raises:
            Error: If `path` is not a valid RFC 9535 query. Section 2.1
                requires an implementation to reject an invalid query
                rather than guess at a meaning for it.
        """
        var q = JSONPath()
        var s = _Scan(path)
        if s.peek() != _C_DOLLAR:
            raise s.error("a query must begin with '$'")
        s.pos += 1
        q.root = q._p_query_tail(s, True)
        if not s.at_end():
            raise s.error("unexpected trailing input")
        return q^

    def query(self, document: Value) raises -> List[Value]:
        """The values of the nodes this query selects, in order."""
        var nodes = self._run(document)
        var out = List[Value]()
        for i in range(len(nodes)):
            out.append(nodes[i].value.copy())
        return out^

    def paths(self, document: Value) raises -> List[String]:
        """The normalized paths of the nodes this query selects.

        A normalized path (RFC 9535 section 2.7) is itself a query, in
        a canonical form that names exactly one node: `$['a'][0]`. It
        is not a JSON Pointer, and the two must not be swapped for each
        other.
        """
        var nodes = self._run(document)
        var out = List[String]()
        for i in range(len(nodes)):
            out.append(nodes[i].path.copy())
        return out^

    def query_with_paths(
        self, document: Value
    ) raises -> List[Tuple[String, Value]]:
        """Each selected node as its normalized path and its value."""
        var nodes = self._run(document)
        var out = List[Tuple[String, Value]]()
        for i in range(len(nodes)):
            out.append((nodes[i].path.copy(), nodes[i].value.copy()))
        return out^

    # -- parsing ---------------------------------------------------

    def _p_query_tail(mut self, mut s: _Scan, from_root: Bool) raises -> Int:
        """Parse `*(S segment)`, the identifier already consumed.

        Whitespace before a segment is consumed only once a segment is
        known to follow. That is what makes `"$.a "` invalid: the ABNF
        has no trailing `S`, so the space is left for `compile` to
        complain about.
        """
        var segs = List[_Segment]()
        while True:
            var save = s.pos
            s.skip_ws()
            var c = s.peek()
            if c != _C_DOT and c != _C_LBRACKET:
                s.pos = save
                break
            segs.append(self._p_segment(s))

        var start = len(self.segments)
        var singular = True
        for i in range(len(segs)):
            var seg = segs[i].copy()
            if seg.descendant or seg.sel_count != 1:
                singular = False
            else:
                var kind = self.selectors[seg.sel_start].kind
                if kind != _SEL_NAME and kind != _SEL_INDEX:
                    singular = False
            self.segments.append(seg^)
        self.queries.append(_QueryRef(start, len(segs), from_root, singular))
        return len(self.queries) - 1

    def _p_segment(mut self, mut s: _Scan) raises -> _Segment:
        if s.peek() == _C_LBRACKET:
            var sels = self._p_bracketed(s)
            return self._commit(sels^, False)

        # A `.` or `..`. Neither admits whitespace before what follows,
        # so the next byte decides on its own.
        s.pos += 1
        var descendant = False
        if s.peek() == _C_DOT:
            s.pos += 1
            descendant = True

        var c = s.peek()
        if c == _C_LBRACKET:
            if not descendant:
                raise s.error("'.' must be followed by a member name or '*'")
            var sels = self._p_bracketed(s)
            return self._commit(sels^, True)
        if c == _C_STAR:
            s.pos += 1
            var sels = List[_Selector]()
            sels.append(_Selector(_SEL_WILDCARD))
            return self._commit(sels^, descendant)
        if descendant and not _is_name_first(c):
            # `$..` on its own selects nothing and means nothing; the
            # ABNF requires a selector after the two dots.
            raise s.error("'..' must be followed by a selector")
        var sels = List[_Selector]()
        sels.append(_Selector(_SEL_NAME, s.member_name()))
        return self._commit(sels^, descendant)

    def _commit(
        mut self, var sels: List[_Selector], descendant: Bool
    ) -> _Segment:
        """Move a segment's selectors into the pool as one run.

        They are collected locally first because parsing a filter
        appends the selectors of its own nested queries, which would
        otherwise interleave with this segment's and break the run.
        """
        var start = len(self.selectors)
        var count = len(sels)
        for i in range(count):
            self.selectors.append(sels[i].copy())
        return _Segment(descendant, start, count)

    def _p_bracketed(mut self, mut s: _Scan) raises -> List[_Selector]:
        """Parse `"[" S selector *(S "," S selector) S "]"`."""
        s.depth += 1
        if s.depth > _MAX_QUERY_DEPTH:
            raise s.error("query nests too deeply")
        s.pos += 1
        var out = List[_Selector]()
        while True:
            s.skip_ws()
            out.append(self._p_selector(s))
            s.skip_ws()
            var c = s.peek()
            if c == _C_COMMA:
                s.pos += 1
                continue
            if c == _C_RBRACKET:
                s.pos += 1
                break
            raise s.error("expected ',' or ']' in a bracketed selection")
        s.depth -= 1
        return out^

    def _p_selector(mut self, mut s: _Scan) raises -> _Selector:
        var c = s.peek()
        if c == _C_STAR:
            s.pos += 1
            return _Selector(_SEL_WILDCARD)
        if c == _C_DQUOTE or c == _C_SQUOTE:
            return _Selector(_SEL_NAME, s.parse_string())
        if c == _C_QUESTION:
            s.pos += 1
            s.skip_ws()
            return _Selector(_SEL_FILTER, String(), self._p_logical_or(s))

        # What is left is an index or a slice, and they are told apart
        # by whether a colon follows the optional first integer.
        var has_start = False
        var start = 0
        if c == _C_MINUS or _is_digit(c):
            start = s.parse_int()
            has_start = True
        var save = s.pos
        s.skip_ws()
        if s.peek() != _C_COLON:
            s.pos = save
            if not has_start:
                raise s.error("expected a selector")
            return _Selector(_SEL_INDEX, String(), start)

        s.pos += 1
        s.skip_ws()
        var has_end = False
        var end = 0
        var d = s.peek()
        if d == _C_MINUS or _is_digit(d):
            end = s.parse_int()
            has_end = True
        var after_end = s.pos
        s.skip_ws()
        var step = 1
        if s.peek() == _C_COLON:
            s.pos += 1
            s.skip_ws()
            var f = s.peek()
            if f == _C_MINUS or _is_digit(f):
                step = s.parse_int()
        else:
            s.pos = after_end
        return _Selector(
            _SEL_SLICE, String(), start, end, step, has_start, has_end
        )

    def _p_logical_or(mut self, mut s: _Scan) raises -> Int:
        var left = self._p_logical_and(s)
        while True:
            var save = s.pos
            s.skip_ws()
            if s.peek() == _C_PIPE and s.peek_at(1) == _C_PIPE:
                s.pos += 2
                s.skip_ws()
                var right = self._p_logical_and(s)
                self.exprs.append(_Expr(_EX_OR, left, right, 0))
                left = len(self.exprs) - 1
            else:
                s.pos = save
                return left

    def _p_logical_and(mut self, mut s: _Scan) raises -> Int:
        var left = self._p_basic(s)
        while True:
            var save = s.pos
            s.skip_ws()
            if s.peek() == _C_AMP and s.peek_at(1) == _C_AMP:
                s.pos += 2
                s.skip_ws()
                var right = self._p_basic(s)
                self.exprs.append(_Expr(_EX_AND, left, right, 0))
                left = len(self.exprs) - 1
            else:
                s.pos = save
                return left

    def _p_basic(mut self, mut s: _Scan) raises -> Int:
        """Parse `paren-expr / comparison-expr / test-expr`.

        The three are not distinguished by their first token, so the
        operand is parsed first and classified afterwards by what
        follows it. That is also where the section 2.3.5.2.1 rule is
        enforced: an operand only has to be a singular query if a
        comparison operator turns up after it.
        """
        var negate = False
        if s.peek() == _C_BANG:
            s.pos += 1
            s.skip_ws()
            negate = True

        if s.peek() == _C_LPAREN:
            s.pos += 1
            s.skip_ws()
            var inner = self._p_logical_or(s)
            s.skip_ws()
            if s.peek() != _C_RPAREN:
                raise s.error("expected ')'")
            s.pos += 1
            if not negate:
                return inner
            self.exprs.append(_Expr(_EX_NOT, inner, 0, 0))
            return len(self.exprs) - 1

        var left = self._p_operand(s)
        if negate:
            # `logical-not-op` attaches to a paren-expr or a test-expr
            # only, so a comparison is not allowed to follow here and
            # the caller will reject whatever does.
            self.exprs.append(_Expr(_EX_NOT, self._as_test(s, left), 0, 0))
            return len(self.exprs) - 1

        var save = s.pos
        s.skip_ws()
        var op = s.comparison_op()
        if op < 0:
            s.pos = save
            return self._as_test(s, left)

        s.skip_ws()
        var right = self._p_operand(s)
        var lhs = self._as_comparable(s, left)
        var rhs = self._as_comparable(s, right)
        self.exprs.append(_Expr(_EX_COMPARE, lhs, rhs, op))
        return len(self.exprs) - 1

    def _as_test(mut self, mut s: _Scan, operand: _Operand) raises -> Int:
        """Use an operand as a `test-expr`, per section 2.4.2."""
        if operand.kind == _OPD_QUERY:
            self.exprs.append(_Expr(_EX_TEST, operand.index, 0, 0))
            return len(self.exprs) - 1
        if operand.kind == _OPD_FUNC:
            if operand.result == _TY_VALUE:
                raise s.error(
                    "a function returning a value is not a test expression"
                )
            self.exprs.append(_Expr(_EX_FUNC_TEST, operand.index, 0, 0))
            return len(self.exprs) - 1
        raise s.error("a literal is not a test expression")

    def _as_comparable(mut self, mut s: _Scan, operand: _Operand) raises -> Int:
        """Store an operand as a comparison operand, checking its type."""
        if operand.kind == _OPD_QUERY and not operand.singular:
            raise s.error("only a singular query may appear in a comparison")
        if operand.kind == _OPD_FUNC and operand.result != _TY_VALUE:
            raise s.error(
                "a function that does not return a value may not be compared"
            )
        self.cmps.append(_Slot(operand.kind, operand.index))
        return len(self.cmps) - 1

    def _p_operand(mut self, mut s: _Scan) raises -> _Operand:
        var c = s.peek()
        if c == _C_AT or c == _C_DOLLAR:
            s.pos += 1
            var q = self._p_query_tail(s, c == _C_DOLLAR)
            return _Operand(_OPD_QUERY, q, self.queries[q].singular, _TY_NODES)
        if c == _C_DQUOTE or c == _C_SQUOTE:
            return self._literal(Value(s.parse_string()))
        if c == _C_MINUS or _is_digit(c):
            return self._literal(s.parse_number())
        if s.word("true"):
            return self._literal(Value(True))
        if s.word("false"):
            return self._literal(Value(False))
        if s.word("null"):
            return self._literal(Value(Null()))
        if _is_lower_alpha(c):
            return self._p_function(s)
        raise s.error("expected a filter operand")

    def _literal(mut self, var value: Value) -> _Operand:
        self.literals.append(value^)
        return _Operand(_OPD_LITERAL, len(self.literals) - 1, True, _TY_VALUE)

    def _p_function(mut self, mut s: _Scan) raises -> _Operand:
        """Parse `function-expr` and check it against its declaration.

        Section 2.4.1 gives every function a parameter list and a
        result type, and section 2.4.3 makes a call that does not match
        them invalid rather than merely useless. An unknown name is
        rejected here too: section 2.4.1 reserves the namespace, so a
        query naming a function this implementation does not have is
        not a query it may answer.
        """
        var start = s.pos
        while _is_func_name_char(s.peek()):
            s.pos += 1
        var name = s.slice(start, s.pos)
        var id = _function_id(name)
        if id < 0:
            raise s.error("unknown function '" + name + "'")
        if s.peek() != _C_LPAREN:
            raise s.error("expected '(' after function name '" + name + "'")
        s.pos += 1

        var params = _function_params(id)
        var arity = len(params)
        var local = List[_Slot]()
        s.skip_ws()
        if s.peek() == _C_RPAREN:
            s.pos += 1
        else:
            while True:
                if len(local) >= arity:
                    raise s.error(
                        "'" + name + "' takes " + String(arity) + " argument(s)"
                    )
                local.append(self._p_argument(s, params[len(local)], name))
                s.skip_ws()
                if s.peek() == _C_COMMA:
                    s.pos += 1
                    s.skip_ws()
                    continue
                if s.peek() == _C_RPAREN:
                    s.pos += 1
                    break
                raise s.error("expected ',' or ')' in a call to '" + name + "'")
        if len(local) != arity:
            raise s.error(
                "'" + name + "' takes " + String(arity) + " argument(s)"
            )

        var arg_start = len(self.args)
        for i in range(arity):
            self.args.append(local[i].copy())
        var compiled = -1
        if id == _FN_MATCH or id == _FN_SEARCH:
            compiled = self._precompile(arg_start + 1)
        self.funcs.append(_Func(id, arg_start, arity, compiled))
        return _Operand(
            _OPD_FUNC, len(self.funcs) - 1, False, _function_result(id)
        )

    def _p_argument(
        mut self, mut s: _Scan, declared: Int, name: String
    ) raises -> _Slot:
        if declared == _TY_NODES:
            var c = s.peek()
            if c == _C_AT or c == _C_DOLLAR:
                s.pos += 1
                return _Slot(_OPD_QUERY, self._p_query_tail(s, c == _C_DOLLAR))
            if _is_lower_alpha(c):
                var f = self._p_function(s)
                if f.result != _TY_NODES:
                    raise s.error(
                        "'" + name + "' needs a query as its argument"
                    )
                return _Slot(_OPD_FUNC, f.index)
            raise s.error("'" + name + "' needs a query as its argument")

        if declared == _TY_LOGICAL:
            return _Slot(_OPD_LOGICAL, self._p_logical_or(s))

        var operand = self._p_operand(s)
        if operand.kind == _OPD_QUERY and not operand.singular:
            raise s.error(
                "'" + name + "' needs a value, so its query must be singular"
            )
        if operand.kind == _OPD_FUNC and operand.result != _TY_VALUE:
            raise s.error(
                "'"
                + name
                + "' needs a value, which '"
                + name
                + "' was not given"
            )
        return _Slot(operand.kind, operand.index)

    def _precompile(mut self, slot: Int) raises -> Int:
        """Compile a literal `match`/`search` pattern ahead of time.

        Reports -1 when the pattern is computed from the document and
        so cannot be compiled yet, and -2 when it is a literal that is
        not a valid I-Regexp and therefore matches nothing.
        """
        var arg = self.args[slot].copy()
        if arg.kind != _OPD_LITERAL:
            return -1
        var literal = self.literals[arg.index].copy()
        if not literal.is_string():
            return -1
        try:
            self.regexes.append(Regex.compile(literal.string_value()))
        except:
            return -2
        return len(self.regexes) - 1

    # -- evaluation ------------------------------------------------

    def _run(self, document: Value) raises -> List[_Node]:
        if self.root < 0:
            raise Error("JSONPath: query was never compiled")
        var root = _Node(String("$"), document.copy())
        return self._eval_query(self.root, document, root, 0)

    def _eval_query(
        self, qi: Int, root: Value, current: _Node, depth: Int
    ) raises -> List[_Node]:
        if depth > _MAX_EVAL_DEPTH:
            raise Error("JSONPath: query nests too deeply to evaluate")
        var q = self.queries[qi].copy()
        var nodes = List[_Node]()
        if q.from_root:
            nodes.append(_Node(String("$"), root.copy()))
        else:
            nodes.append(current.copy())
        for i in range(q.seg_count):
            nodes = self._apply_segment(
                self.segments[q.seg_start + i], root, nodes^, depth
            )
        return nodes^

    def _apply_segment(
        self,
        seg: _Segment,
        root: Value,
        var input: List[_Node],
        depth: Int,
    ) raises -> List[_Node]:
        """Apply one segment to a nodelist.

        Section 2.5.1.2 orders the result by input node first and by
        selector second, and the descendant segment of section 2.5.2.2
        keeps that ordering with the input node and its descendants
        standing in for the single input node.
        """
        var out = List[_Node]()
        for i in range(len(input)):
            if seg.descendant:
                var family = self._descendants(input[i])
                for j in range(len(family)):
                    for k in range(seg.sel_count):
                        self._apply_selector(
                            self.selectors[seg.sel_start + k],
                            root,
                            family[j],
                            out,
                            depth,
                        )
            else:
                for k in range(seg.sel_count):
                    self._apply_selector(
                        self.selectors[seg.sel_start + k],
                        root,
                        input[i],
                        out,
                        depth,
                    )
        return out^

    def _apply_selector(
        self,
        sel: _Selector,
        root: Value,
        node: _Node,
        mut out: List[_Node],
        depth: Int,
    ) raises:
        if sel.kind == _SEL_NAME:
            if node.value.is_object():
                var member = _member(node.value, sel.name)
                # An object without that member contributes no node,
                # which is an empty result rather than an error.
                if member.present:
                    out.append(
                        _Node(
                            _path_member(node.path, sel.name),
                            member.value.copy(),
                        )
                    )
            return

        if sel.kind == _SEL_WILDCARD:
            if node.value.is_array():
                var items = node.value.array_items()
                for i in range(len(items)):
                    out.append(
                        _Node(_path_index(node.path, i), items[i].copy())
                    )
            elif node.value.is_object():
                var members = node.value.object_items()
                for i in range(len(members)):
                    out.append(
                        _Node(
                            _path_member(node.path, members[i][0]),
                            members[i][1].copy(),
                        )
                    )
            return

        if sel.kind == _SEL_INDEX:
            if node.value.is_array():
                var count = node.value.array_count()
                # Resolved against this node's length, every time. The
                # old code hoisted this out of the loop, so the first
                # array's length decided the index for every array
                # after it.
                var index = sel.a
                if index < 0:
                    index = count + index
                if index >= 0 and index < count:
                    out.append(
                        _Node(
                            _path_index(node.path, index),
                            node.value[index],
                        )
                    )
            return

        if sel.kind == _SEL_SLICE:
            if node.value.is_array():
                self._apply_slice(sel, node, out)
            return

        # A filter selector. Section 2.3.5.2 applies it to the elements
        # of an array and to the member values of an object alike.
        if node.value.is_array():
            var items = node.value.array_items()
            for i in range(len(items)):
                var child = _Node(_path_index(node.path, i), items[i].copy())
                if self._eval_logical(sel.a, root, child, depth + 1):
                    out.append(child^)
        elif node.value.is_object():
            var members = node.value.object_items()
            for i in range(len(members)):
                var child = _Node(
                    _path_member(node.path, members[i][0]),
                    members[i][1].copy(),
                )
                if self._eval_logical(sel.a, root, child, depth + 1):
                    out.append(child^)

    def _apply_slice(
        self, sel: _Selector, node: _Node, mut out: List[_Node]
    ) raises:
        """The slice of section 2.3.4.2.2, bounds and all.

        Written to follow the RFC's own pseudocode rather than to look
        like a Python slice, because the two differ at the edges and
        the RFC is the one being implemented.
        """
        var step = sel.c
        if step == 0:
            # Section 2.3.4.2.2 says a zero step selects nothing. The
            # old code advanced by zero and appended until it ran out
            # of memory.
            return
        var items = node.value.array_items()
        var count = len(items)

        var start = sel.a
        if not sel.has_a:
            start = 0 if step >= 0 else count - 1
        var end = sel.b
        if not sel.has_b:
            # For a negative step the end has to fall before element
            # zero, and `-count - 1` is the value that normalizes to
            # -1 for an array of any length.
            end = count if step >= 0 else -count - 1
        if start < 0:
            start = count + start
        if end < 0:
            end = count + end

        if step > 0:
            var lower = min(max(start, 0), count)
            var upper = min(max(end, 0), count)
            var i = lower
            while i < upper:
                out.append(_Node(_path_index(node.path, i), items[i].copy()))
                i += step
        else:
            var upper = min(max(start, -1), count - 1)
            var lower = min(max(end, -1), count - 1)
            var i = upper
            while lower < i:
                out.append(_Node(_path_index(node.path, i), items[i].copy()))
                i += step

    def _descendants(self, node: _Node) raises -> List[_Node]:
        """A node and all its descendants, in document order.

        Iterative rather than recursive: the document's depth is not
        bounded by anything the query says, and a walk that recurses
        once per level turns a deeply nested document into a crash
        instead of an answer.
        """
        var out = List[_Node]()
        var stack = List[_Node]()
        var depths = List[Int]()
        stack.append(node.copy())
        depths.append(0)
        while len(stack) > 0:
            var top = stack.pop()
            var level = depths.pop()
            if level > _MAX_DESCENT_DEPTH:
                raise Error(
                    "JSONPath: document nests deeper than a descendant"
                    " segment will walk"
                )
            if top.value.is_array():
                var items = top.value.array_items()
                # Pushed in reverse so that popping yields document
                # order.
                for k in range(len(items) - 1, -1, -1):
                    stack.append(
                        _Node(_path_index(top.path, k), items[k].copy())
                    )
                    depths.append(level + 1)
            elif top.value.is_object():
                var members = top.value.object_items()
                for k in range(len(members) - 1, -1, -1):
                    stack.append(
                        _Node(
                            _path_member(top.path, members[k][0]),
                            members[k][1].copy(),
                        )
                    )
                    depths.append(level + 1)
            out.append(top^)
        return out^

    def _eval_logical(
        self, ei: Int, root: Value, current: _Node, depth: Int
    ) raises -> Bool:
        if depth > _MAX_EVAL_DEPTH:
            raise Error("JSONPath: filter nests too deeply to evaluate")
        var e = self.exprs[ei].copy()
        if e.kind == _EX_OR:
            if self._eval_logical(e.a, root, current, depth):
                return True
            return self._eval_logical(e.b, root, current, depth)
        if e.kind == _EX_AND:
            if not self._eval_logical(e.a, root, current, depth):
                return False
            return self._eval_logical(e.b, root, current, depth)
        if e.kind == _EX_NOT:
            return not self._eval_logical(e.a, root, current, depth)
        if e.kind == _EX_TEST:
            # An existence test: a query is true when it selects at
            # least one node, whatever those nodes hold.
            var nodes = self._eval_query(e.a, root, current, depth + 1)
            return len(nodes) > 0
        if e.kind == _EX_FUNC_TEST:
            return self._eval_logical_func(e.a, root, current, depth + 1)
        var lhs = self._eval_comparable(e.a, root, current, depth + 1)
        var rhs = self._eval_comparable(e.b, root, current, depth + 1)
        return _compare(e.op, lhs, rhs)

    def _eval_comparable(
        self, ci: Int, root: Value, current: _Node, depth: Int
    ) raises -> _Maybe:
        var slot = self.cmps[ci].copy()
        return self._eval_slot(slot, root, current, depth)

    def _eval_slot(
        self, slot: _Slot, root: Value, current: _Node, depth: Int
    ) raises -> _Maybe:
        """An operand read as a value, or as Nothing."""
        if slot.kind == _OPD_LITERAL:
            return _Maybe(True, self.literals[slot.index].copy())
        if slot.kind == _OPD_QUERY:
            # The parser has already proved this query is singular, so
            # it selects at most one node and the absent case is the
            # Nothing of section 2.4.1.
            var nodes = self._eval_query(slot.index, root, current, depth)
            if len(nodes) == 1:
                return _Maybe(True, nodes[0].value.copy())
            return _nothing()
        return self._eval_value_func(slot.index, root, current, depth)

    def _eval_value_func(
        self, fi: Int, root: Value, current: _Node, depth: Int
    ) raises -> _Maybe:
        var f = self.funcs[fi].copy()
        if f.id == _FN_LENGTH:
            var arg = self._eval_slot(
                self.args[f.arg_start], root, current, depth
            )
            if not arg.present:
                return _nothing()
            if arg.value.is_string():
                # Section 2.4.4 counts Unicode scalar values, not the
                # bytes that encode them.
                return _Maybe(
                    True, Value(_codepoint_count(arg.value.string_value()))
                )
            if arg.value.is_array():
                return _Maybe(True, Value(arg.value.array_count()))
            if arg.value.is_object():
                return _Maybe(True, Value(arg.value.object_count()))
            return _nothing()
        if f.id == _FN_COUNT:
            var nodes = self._eval_nodes_arg(f.arg_start, root, current, depth)
            return _Maybe(True, Value(len(nodes)))
        if f.id == _FN_VALUE:
            var nodes = self._eval_nodes_arg(f.arg_start, root, current, depth)
            if len(nodes) == 1:
                return _Maybe(True, nodes[0].value.copy())
            return _nothing()
        raise Error("JSONPath: function does not return a value")

    def _eval_logical_func(
        self, fi: Int, root: Value, current: _Node, depth: Int
    ) raises -> Bool:
        var f = self.funcs[fi].copy()
        if f.id != _FN_MATCH and f.id != _FN_SEARCH:
            raise Error("JSONPath: function does not return a logical value")
        var subject = self._eval_slot(
            self.args[f.arg_start], root, current, depth
        )
        if not subject.present or not subject.value.is_string():
            return False
        var text = subject.value.string_value()

        if f.regex >= 0:
            if f.id == _FN_MATCH:
                return self.regexes[f.regex].full_match(text)
            return self.regexes[f.regex].search(text)
        if f.regex == -2:
            # A literal pattern that does not compile matches nothing.
            return False

        var pattern = self._eval_slot(
            self.args[f.arg_start + 1], root, current, depth
        )
        if not pattern.present or not pattern.value.is_string():
            return False
        try:
            var compiled = Regex.compile(pattern.value.string_value())
            if f.id == _FN_MATCH:
                return compiled.full_match(text)
            return compiled.search(text)
        except:
            return False

    def _eval_nodes_arg(
        self, slot: Int, root: Value, current: _Node, depth: Int
    ) raises -> List[_Node]:
        var arg = self.args[slot].copy()
        if arg.kind == _OPD_QUERY:
            return self._eval_query(arg.index, root, current, depth)
        raise Error("JSONPath: no function extension here returns a nodelist")


def jsonpath_query(document: Value, path: String) raises -> List[Value]:
    """Query a JSON document with an RFC 9535 JSONPath expression.

    Args:
        document: The document to query.
        path: The query, for example `$.store.book[?@.price < 10].title`.

    Returns:
        The values of the selected nodes, in the order RFC 9535 gives
        them.

    Raises:
        Error: If `path` is not a valid query.

    Example:
        var doc = loads('{"users":[{"name":"Alice"},{"name":"Bob"}]}')
        var names = jsonpath_query(doc, "$.users[*].name")
        Returns `[Value("Alice"), Value("Bob")]`.
    """
    return JSONPath.compile(path).query(document)


def jsonpath_one(document: Value, path: String) raises -> Value:
    """The first node a query selects.

    Args:
        document: The document to query.
        path: The query.

    Returns:
        The value of the first selected node.

    Raises:
        Error: If `path` is not a valid query, or selects no node.
    """
    var results = JSONPath.compile(path).query(document)
    if len(results) == 0:
        raise Error("No match found for JSONPath: " + path)
    return results[0].copy()


# -- normalized paths (RFC 9535 section 2.7) -----------------------


def _path_index(base: String, index: Int) -> String:
    return base + "[" + String(index) + "]"


def _path_member(base: String, name: String) -> String:
    return base + "['" + _escape_normal(name) + "']"


def _escape_normal(name: String) -> String:
    """A member name as `normal-single-quoted` from section 2.7.

    Only the apostrophe, the reverse solidus and the controls are
    escaped. The solidus and the quotation mark stand for themselves
    here, unlike in a JSON string, and the hexadecimal digits of a
    `\\u` escape are lowercase.
    """
    var out = List[UInt8]()
    var bytes = name.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(_C_BACKSLASH) or c == UInt8(_C_SQUOTE):
            out.append(UInt8(_C_BACKSLASH))
            out.append(c)
        elif c == 0x08:
            out.append(UInt8(_C_BACKSLASH))
            out.append(UInt8(_C_LOWER_B))
        elif c == 0x09:
            out.append(UInt8(_C_BACKSLASH))
            out.append(UInt8(_C_LOWER_T))
        elif c == 0x0A:
            out.append(UInt8(_C_BACKSLASH))
            out.append(UInt8(_C_LOWER_N))
        elif c == 0x0C:
            out.append(UInt8(_C_BACKSLASH))
            out.append(UInt8(_C_LOWER_F))
        elif c == 0x0D:
            out.append(UInt8(_C_BACKSLASH))
            out.append(UInt8(_C_LOWER_R))
        elif c < 0x20:
            out.append(UInt8(_C_BACKSLASH))
            out.append(UInt8(_C_LOWER_U))
            out.append(UInt8(_C_0))
            out.append(UInt8(_C_0))
            out.append(_hex_digit(Int(c) >> 4))
            out.append(_hex_digit(Int(c) & 0x0F))
        else:
            out.append(c)
    return String(unsafe_from_utf8=Span(out))


def _hex_digit(value: Int) -> UInt8:
    if value < 10:
        return UInt8(_C_0 + value)
    return UInt8(_C_LOWER_A + value - 10)


# -- comparison (RFC 9535 section 2.3.5.2.2) -----------------------


def _compare(op: Int, a: _Maybe, b: _Maybe) raises -> Bool:
    if op == _OP_EQ:
        return _values_equal(a, b)
    if op == _OP_NE:
        return not _values_equal(a, b)
    if op == _OP_LT:
        return _value_less(a, b)
    if op == _OP_LE:
        return _value_less(a, b) or _values_equal(a, b)
    if op == _OP_GT:
        return _value_less(b, a)
    return _value_less(b, a) or _values_equal(a, b)


def _values_equal(a: _Maybe, b: _Maybe) raises -> Bool:
    """Equality where Nothing is a value of its own.

    Nothing equals Nothing and nothing else, which is what makes
    `@.missing == @.other_missing` true and `@.missing != 1` true at
    the same time.
    """
    if not a.present or not b.present:
        return a.present == b.present
    return _deep_equal(a.value, b.value, 0)


def _value_less(a: _Maybe, b: _Maybe) -> Bool:
    """Strict ordering, which only two numbers or two strings have.

    Section 2.3.5.2.2 gives `<` a meaning for those two pairs and makes
    it false everywhere else. The old code returned "equal" for
    incomparable operands, so `@.name <= 5` held for every object with
    a name.
    """
    if not a.present or not b.present:
        return False
    if a.value.is_number() and b.value.is_number():
        return _number_order(a.value, b.value) < 0
    if a.value.is_string() and b.value.is_string():
        # UTF-8 orders by byte exactly as Unicode orders by code point,
        # so a byte comparison is the code point comparison the RFC
        # asks for.
        return a.value.string_value() < b.value.string_value()
    return False


def _deep_equal(a: Value, b: Value, depth: Int) raises -> Bool:
    """Structural equality, ignoring member order.

    `Value.__eq__` compares serialized text, which calls `{"a":1,"b":2}`
    and `{"b":2,"a":1}` different and `1` and `1.0` different. Both
    readings are wrong for section 2.3.5.2.2, so this walks the two
    values instead.
    """
    if depth > _MAX_EQUAL_DEPTH:
        raise Error("JSONPath: values nest too deeply to compare")
    if a.is_null():
        return b.is_null()
    if a.is_bool():
        return b.is_bool() and a.bool_value() == b.bool_value()
    if a.is_number():
        return b.is_number() and _number_order(a, b) == 0
    if a.is_string():
        return b.is_string() and a.string_value() == b.string_value()
    if a.is_array():
        if not b.is_array():
            return False
        var left = a.array_items()
        var right = b.array_items()
        if len(left) != len(right):
            return False
        for i in range(len(left)):
            if not _deep_equal(left[i], right[i], depth + 1):
                return False
        return True
    if a.is_object():
        if not b.is_object():
            return False
        var left = a.object_items()
        if len(left) != b.object_count():
            return False
        for i in range(len(left)):
            var other = _member(b, left[i][0])
            # A member of `a` that `b` does not have. The two have the
            # same member count, so that settles it.
            if not other.present:
                return False
            if not _deep_equal(left[i][1], other.value, depth + 1):
                return False
        return True
    return False


def _member(value: Value, name: String) raises -> _Maybe:
    """The member of an object named `name`, or Nothing.

    Scanning rather than calling `Value.__getitem__` keeps a missing
    member from travelling as an exception, which it is not: a name
    selector that matches nothing simply contributes no node.
    """
    var members = value.object_items()
    for i in range(len(members)):
        if members[i][0] == name:
            return _Maybe(True, members[i][1].copy())
    return _nothing()


def _number_order(a: Value, b: Value) -> Int:
    """Order two JSON numbers by value, across their representations.

    Integers are compared as integers wherever both sides are stored
    that way, so two values more than `2**53` apart in magnitude do not
    collapse onto the same double and compare equal.
    """
    if a.is_int() and b.is_int():
        var ai = a.int_value()
        var bi = b.int_value()
        if ai < bi:
            return -1
        if ai > bi:
            return 1
        return 0
    if a.is_uint() and b.is_uint():
        var au = a.uint_value()
        var bu = b.uint_value()
        if au < bu:
            return -1
        if au > bu:
            return 1
        return 0
    # `is_uint` means a magnitude above `Int64.MAX`, so an unsigned
    # value always outranks a signed one.
    if a.is_uint() and b.is_int():
        return 1
    if a.is_int() and b.is_uint():
        return -1
    var af = _as_float(a)
    var bf = _as_float(b)
    if af < bf:
        return -1
    if af > bf:
        return 1
    return 0


def _as_float(v: Value) -> Float64:
    if v.is_float():
        return v.float_value()
    if v.is_uint():
        return Float64(v.uint_value())
    return Float64(v.int_value())


# -- small lexical helpers -----------------------------------------


def _is_digit(c: Int) -> Bool:
    return c >= _C_0 and c <= _C_9


def _is_lower_alpha(c: Int) -> Bool:
    return c >= _C_LOWER_A and c <= _C_LOWER_Z


def _is_name_first(c: Int) -> Bool:
    if c >= _C_UPPER_A and c <= _C_UPPER_Z:
        return True
    if c >= _C_LOWER_A and c <= _C_LOWER_Z:
        return True
    if c == _C_UNDERSCORE:
        return True
    return c >= 0x80


def _is_name_char(c: Int) -> Bool:
    return _is_name_first(c) or _is_digit(c)


def _is_func_name_char(c: Int) -> Bool:
    return _is_lower_alpha(c) or _is_digit(c) or c == _C_UNDERSCORE


def _hex_value(c: Int) -> Int:
    if c >= _C_0 and c <= _C_9:
        return c - _C_0
    if c >= 0x61 and c <= 0x66:
        return c - 0x61 + 10
    if c >= 0x41 and c <= 0x46:
        return c - 0x41 + 10
    return -1


def _encode_utf8(code: Int, mut out: List[UInt8]):
    if code < 0x80:
        out.append(UInt8(code))
    elif code < 0x800:
        out.append(UInt8(0xC0 | (code >> 6)))
        out.append(UInt8(0x80 | (code & 0x3F)))
    elif code < 0x10000:
        out.append(UInt8(0xE0 | (code >> 12)))
        out.append(UInt8(0x80 | ((code >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (code & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (code >> 18)))
        out.append(UInt8(0x80 | ((code >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((code >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (code & 0x3F)))


def _codepoint_count(s: String) -> Int:
    """The number of Unicode scalar values in `s`.

    Every byte that is not a UTF-8 continuation byte starts exactly one
    scalar value, so counting those counts the characters without
    decoding any of them.
    """
    var bytes = s.as_bytes()
    var count = 0
    for i in range(len(bytes)):
        if (bytes[i] & 0xC0) != 0x80:
            count += 1
    return count


# -- the function extension table (RFC 9535 section 2.4) -----------


def _function_id(name: String) -> Int:
    if name == "length":
        return _FN_LENGTH
    if name == "count":
        return _FN_COUNT
    if name == "match":
        return _FN_MATCH
    if name == "search":
        return _FN_SEARCH
    if name == "value":
        return _FN_VALUE
    return -1


def _function_params(id: Int) -> List[Int]:
    var params = List[Int]()
    if id == _FN_LENGTH:
        params.append(_TY_VALUE)
    elif id == _FN_COUNT:
        params.append(_TY_NODES)
    elif id == _FN_VALUE:
        params.append(_TY_NODES)
    else:
        params.append(_TY_VALUE)
        params.append(_TY_VALUE)
    return params^


def _function_result(id: Int) -> Int:
    if id == _FN_MATCH or id == _FN_SEARCH:
        return _TY_LOGICAL
    return _TY_VALUE
