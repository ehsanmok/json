# json - JSON Schema validation (draft 2020-12).
#
# The previous version of this module advertised a draft-07 subset and
# implemented twenty keywords, several of them incorrectly. The three
# failures that mattered most were silent ones: `$ref` was ignored, so
# every schema built out of `$defs` validated everything; `pattern` was
# a literal string comparison dressed up as a regular expression, so
# most patterns validated everything; and every keyword sat inside a
# bare `try: ... except: pass` used as a "key absent" test, so a
# malformed schema or a raise from deep inside recursion also read as
# "constraint satisfied". A validator that answers "valid" when it does
# not know is worse than one that has no opinion, because the caller
# cannot tell the two apart.
#
# This version compiles a schema document once into a flat list of
# nodes with integer child indices, resolves `$ref` against that list,
# and validates against the compiled form. Presence of a keyword is
# decided while compiling, by walking the schema object's members, so
# nothing depends on catching an exception to learn whether a member
# exists.
#
# What is deliberately not here:
#
# - `$dynamicRef` and `$dynamicAnchor`. Their resolution depends on the
#   dynamic scope of the evaluation, which means the target of a
#   reference is not known until the instance is being walked. That is
#   incompatible with resolving references once at compile time, and
#   the keywords exist to support recursive extension of a schema
#   across documents, which needs remote retrieval to be useful. A
#   schema using them is rejected at compile time rather than validated
#   with the keyword ignored.
# - Remote references. A `$ref` is resolved inside the schema document
#   it appears in. Anything naming another document is rejected at
#   compile time, because retrieval is a policy decision that belongs
#   to the caller and silently ignoring the reference is the bug this
#   rewrite exists to remove.
# - `contentEncoding`, `contentMediaType` and `contentSchema`. Draft
#   2020-12 defines all three as annotations that never affect
#   validation, so collecting them would add surface without changing
#   an answer.

from std.collections import Dict, List
from std.math import floor

from .pointer import build_pointer, escape_token, parse_pointer
from .regex import Regex
from .value import Null, Value


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

# A `$ref` back to an ancestor is legal and useful, so a cycle cannot
# be rejected at compile time. It is bounded at validation time
# instead: a self-referential schema with no base case exhausts this
# budget and raises, rather than running the stack out and taking the
# process with it. The cap is well past any hand-written schema nesting
# and any plausible document depth.
comptime _MAX_DEPTH = 256

# Type codes. The instance types of draft 2020-12 section 6.1.1, plus
# `integer`, which is a restriction of `number` rather than a separate
# instance type.
comptime _T_NULL = 0
comptime _T_BOOLEAN = 1
comptime _T_OBJECT = 2
comptime _T_ARRAY = 3
comptime _T_NUMBER = 4
comptime _T_STRING = 5
comptime _T_INTEGER = 6


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


struct ValidationError(Copyable, Movable, Writable):
    """One reason an instance failed, and where both ends of it are.

    `path` is a JSON Pointer into the instance and `keyword_location`
    is a JSON Pointer into the schema. Reporting only the first leaves
    the caller guessing which of the keywords covering that location
    objected, which is the common case as soon as a schema uses
    `allOf` or `$ref`.
    """

    var path: String
    var keyword_location: String
    var message: String

    def __init__(out self, path: String, message: String):
        self.path = path
        self.keyword_location = ""
        self.message = message

    def __init__(
        out self, path: String, keyword_location: String, message: String
    ):
        self.path = path
        self.keyword_location = keyword_location
        self.message = message

    def __str__(self) -> String:
        if self.path == "":
            return self.message
        return self.path + ": " + self.message

    def write_to[W: Writer](self, mut writer: W):
        if self.path != "":
            writer.write(self.path, ": ")
        writer.write(self.message)


@fieldwise_init
struct FormatAnnotation(Copyable, Movable):
    """One `format` keyword that was reached, and what it observed.

    Draft 2020-12 section 7.2.1 makes `format` an annotation by
    default. Collecting the observation rather than discarding it is
    what makes the default mode useful: a caller can report a bad
    email address without having turned every `format` in the schema
    into an assertion.
    """

    var path: String
    var format: String
    var matched: Bool


struct ValidationResult(Boolable, Movable):
    """The verdict, every reason for it, and the `format` annotations."""

    var valid: Bool
    var errors: List[ValidationError]
    var format_annotations: List[FormatAnnotation]

    def __init__(out self):
        self.valid = True
        self.errors = List[ValidationError]()
        self.format_annotations = List[FormatAnnotation]()

    def __bool__(self) -> Bool:
        return self.valid

    def add_error(mut self, path: String, message: String):
        self.valid = False
        self.errors.append(ValidationError(path, "", message))

    def add_error(
        mut self, path: String, keyword_location: String, message: String
    ):
        self.valid = False
        self.errors.append(ValidationError(path, keyword_location, message))


# ---------------------------------------------------------------------------
# Compiled schema nodes
# ---------------------------------------------------------------------------


struct _Node(Copyable, Movable):
    """One subschema, with every child reduced to an index.

    Mojo has no recursive structs, and a schema is a tree, so the tree
    is held as a list and a node names its children by position. The
    absent marker for an index or a non-negative bound is -1, which no
    valid operand can take; the keywords whose operand may legitimately
    be any number carry a separate presence flag.
    """

    var location: String
    var is_bool: Bool
    var bool_ok: Bool

    var ref_target: Int

    var types: List[Int]

    var has_enum: Bool
    var enum_values: List[Value]
    var has_const: Bool
    var const_value: Value

    var has_minimum: Bool
    var minimum: Value
    var has_maximum: Bool
    var maximum: Value
    var has_exclusive_minimum: Bool
    var exclusive_minimum: Value
    var has_exclusive_maximum: Bool
    var exclusive_maximum: Value
    var has_multiple_of: Bool
    var multiple_of: Value

    var min_length: Int
    var max_length: Int
    var pattern_index: Int
    var pattern_source: String
    var format_name: String

    var min_items: Int
    var max_items: Int
    var unique_items: Bool
    var prefix_items: List[Int]
    var items_node: Int
    var contains_node: Int
    var min_contains: Int
    var max_contains: Int
    var unevaluated_items: Int

    var min_properties: Int
    var max_properties: Int
    var required: List[String]
    var property_names: List[String]
    var property_nodes: List[Int]
    var pattern_property_regex: List[Int]
    var pattern_property_source: List[String]
    var pattern_property_nodes: List[Int]
    var additional_properties: Int
    var property_names_node: Int
    var dependent_required_keys: List[String]
    var dependent_required_values: List[List[String]]
    var dependent_schema_keys: List[String]
    var dependent_schema_nodes: List[Int]
    var unevaluated_properties: Int

    var all_of: List[Int]
    var has_any_of: Bool
    var any_of: List[Int]
    var has_one_of: Bool
    var one_of: List[Int]
    var not_node: Int
    var if_node: Int
    var then_node: Int
    var else_node: Int

    def __init__(out self, location: String):
        self.location = location
        self.is_bool = False
        self.bool_ok = True
        self.ref_target = -1
        self.types = List[Int]()
        self.has_enum = False
        self.enum_values = List[Value]()
        self.has_const = False
        self.const_value = Value(Null())
        self.has_minimum = False
        self.minimum = Value(Null())
        self.has_maximum = False
        self.maximum = Value(Null())
        self.has_exclusive_minimum = False
        self.exclusive_minimum = Value(Null())
        self.has_exclusive_maximum = False
        self.exclusive_maximum = Value(Null())
        self.has_multiple_of = False
        self.multiple_of = Value(Null())
        self.min_length = -1
        self.max_length = -1
        self.pattern_index = -1
        self.pattern_source = ""
        self.format_name = ""
        self.min_items = -1
        self.max_items = -1
        self.unique_items = False
        self.prefix_items = List[Int]()
        self.items_node = -1
        self.contains_node = -1
        self.min_contains = -1
        self.max_contains = -1
        self.unevaluated_items = -1
        self.min_properties = -1
        self.max_properties = -1
        self.required = List[String]()
        self.property_names = List[String]()
        self.property_nodes = List[Int]()
        self.pattern_property_regex = List[Int]()
        self.pattern_property_source = List[String]()
        self.pattern_property_nodes = List[Int]()
        self.additional_properties = -1
        self.property_names_node = -1
        self.dependent_required_keys = List[String]()
        self.dependent_required_values = List[List[String]]()
        self.dependent_schema_keys = List[String]()
        self.dependent_schema_nodes = List[Int]()
        self.unevaluated_properties = -1
        self.all_of = List[Int]()
        self.has_any_of = False
        self.any_of = List[Int]()
        self.has_one_of = False
        self.one_of = List[Int]()
        self.not_node = -1
        self.if_node = -1
        self.then_node = -1
        self.else_node = -1


struct _Annot(Copyable, Movable):
    """What an applicator evaluated at one instance location.

    `unevaluatedProperties` and `unevaluatedItems` are defined against
    the annotations the other keywords produced, so those annotations
    have to travel back up out of every in-place applicator. They stop
    at the instance location they describe: a subschema applied to a
    child instance reports nothing here, because what it evaluated is
    not a member of the parent.
    """

    var props: List[String]
    var items_upto: Int
    var extra_items: List[Int]

    def __init__(out self):
        self.props = List[String]()
        self.items_upto = 0
        self.extra_items = List[Int]()

    def merge(mut self, other: _Annot):
        for i in range(len(other.props)):
            self.props.append(other.props[i])
        if other.items_upto > self.items_upto:
            self.items_upto = other.items_upto
        for i in range(len(other.extra_items)):
            self.extra_items.append(other.extra_items[i])

    def covers_property(self, name: String) -> Bool:
        for i in range(len(self.props)):
            if self.props[i] == name:
                return True
        return False

    def covers_item(self, index: Int) -> Bool:
        if index < self.items_upto:
            return True
        for i in range(len(self.extra_items)):
            if self.extra_items[i] == index:
                return True
        return False


# ---------------------------------------------------------------------------
# Small helpers over `Value`
# ---------------------------------------------------------------------------


def _type_name(value: Value) -> String:
    """The instance type of `value`, in the spelling the spec uses."""
    if value.is_null():
        return "null"
    if value.is_bool():
        return "boolean"
    if value.is_object():
        return "object"
    if value.is_array():
        return "array"
    if value.is_string():
        return "string"
    return "number"


def _as_float(value: Value) -> Float64:
    """A JSON number widened to binary64, whatever tag it carries."""
    if value.is_float():
        return value.float_value()
    if value.is_uint():
        return Float64(value.uint_value())
    return Float64(value.int_value())


def _is_integral(value: Value) -> Bool:
    """Whether `value` is a number with no fractional part.

    Draft 2020-12 section 6.1.1 says a float with zero fractional part
    is an integer, so `1.0` is one. The previous version tested the
    parser's tag instead, which made the answer depend on how the
    document happened to be spelled.
    """
    if value.is_int() or value.is_uint():
        return True
    if not value.is_float():
        return False
    var f = value.float_value()
    # An infinity or a NaN cannot come out of the JSON grammar, but a
    # caller can build a `Value` by hand, and `floor` of either is not
    # equal to itself for NaN and is equal for an infinity. Neither is
    # an integer for this purpose.
    if f != f:
        return False
    if f - f != 0.0:
        return False
    return floor(f) == f


def _compare_numbers(a: Value, b: Value) -> Int:
    """Three-way comparison of two JSON numbers, exact where it can be.

    Two integers are compared as integers, because widening to binary64
    first would make every pair above 2**53 that differs by one compare
    equal, and `minimum` on an identifier-sized integer is a real use.
    """
    if (a.is_int() or a.is_uint()) and (b.is_int() or b.is_uint()):
        if a.is_int() and b.is_int():
            var ai = a.int_value()
            var bi = b.int_value()
            if ai < bi:
                return -1
            return 1 if ai > bi else 0
        if a.is_uint() and b.is_uint():
            var au = a.uint_value()
            var bu = b.uint_value()
            if au < bu:
                return -1
            return 1 if au > bu else 0
        # Exactly one side is above `Int64.MAX`, so it is the larger
        # unless the other is negative, in which case it is still the
        # larger. The unsigned side always wins.
        if a.is_uint():
            return 1
        return -1

    var af = _as_float(a)
    var bf = _as_float(b)
    if af < bf:
        return -1
    return 1 if af > bf else 0


def _deep_equal(a: Value, b: Value) raises -> Bool:
    """Structural equality as draft 2020-12 section 4.2.2 defines it.

    Member order is not part of an object's identity, and numbers are
    equal when their mathematical values are equal, so `1` equals
    `1.0`. `Value.__eq__` compares serialized text and answers no to
    both of those, which is why `enum`, `const` and `uniqueItems` do
    not use it.
    """
    if a.is_null():
        return b.is_null()
    if a.is_bool():
        return b.is_bool() and a.bool_value() == b.bool_value()
    if a.is_string():
        return b.is_string() and a.string_value() == b.string_value()
    if a.is_number():
        return b.is_number() and _compare_numbers(a, b) == 0
    if a.is_array():
        if not b.is_array():
            return False
        var ai = a.array_items()
        var bi = b.array_items()
        if len(ai) != len(bi):
            return False
        for i in range(len(ai)):
            if not _deep_equal(ai[i], bi[i]):
                return False
        return True
    if a.is_object():
        if not b.is_object():
            return False
        var am = a.object_items()
        var bm = b.object_items()
        if len(am) != len(bm):
            return False
        for i in range(len(am)):
            var found = False
            for j in range(len(bm)):
                if am[i][0] == bm[j][0]:
                    if not _deep_equal(am[i][1], bm[j][1]):
                        return False
                    found = True
                    break
            if not found:
                return False
        return True
    return False


def _code_point_count(s: String) -> Int:
    """The number of Unicode code points in `s`.

    `minLength` and `maxLength` are defined on characters, and a byte
    count answers a different question as soon as the string leaves
    ASCII. Every byte of a well-formed UTF-8 sequence except the first
    matches `10xxxxxx`, so the leading bytes are the count.
    """
    var bytes = s.as_bytes()
    var n = 0
    for i in range(len(bytes)):
        if (bytes[i] & 0xC0) != 0x80:
            n += 1
    return n


def _has_member(obj: Value, key: String) -> Bool:
    """Whether an object has a member named `key`."""
    if not obj.is_object():
        return False
    var keys = obj.object_keys()
    for i in range(len(keys)):
        if keys[i] == key:
            return True
    return False


def _child_path(path: String, token: String) -> String:
    """The instance pointer for a member or element of `path`.

    The tokens are escaped, because a member named `a/b` concatenated
    raw produces a pointer that addresses a different location, and a
    member named `~1` produces one that addresses `/`.
    """
    return path + "/" + escape_token(token)


# ---------------------------------------------------------------------------
# Pattern compilation
# ---------------------------------------------------------------------------

# The engine in `json/regex.mojo` implements I-Regexp (RFC 9485), which
# has no anchors: `^` and `$` are ordinary characters there. JSON
# Schema `pattern` is ECMA-262 and unanchored, so a bare pattern is a
# search. A leading `^` or a trailing `$` is an anchor in ECMA-262 and
# has to be honoured here rather than handed to the engine as a
# literal, which is what the four modes below are for.
comptime _PAT_SEARCH = 0
comptime _PAT_FULL = 1
comptime _PAT_PREFIX = 2
comptime _PAT_SUFFIX = 3


def _strip_prefix(s: String, count: Int) -> String:
    return String(unsafe_from_utf8=s.as_bytes()[count : s.byte_length()])


def _strip_suffix(s: String, count: Int) -> String:
    return String(unsafe_from_utf8=s.as_bytes()[0 : s.byte_length() - count])


def _pattern_program(pattern: String) raises -> Regex:
    """Compile a JSON Schema `pattern` into an I-Regexp program.

    Anchors are folded into the program rather than into the matcher,
    so every pattern can be run with `full_match` and the caller does
    not need a separate mode. `(\\s|\\S)*` is the any-character filler:
    the dot is `[^\\n\\r]` in I-Regexp and would refuse to skip a line
    break.

    Raises:
        Error: If the pattern is outside I-Regexp. ECMA-262 is a
            larger language, so a lookahead or a backreference is
            reported here rather than quietly ignored.
    """
    var body = pattern
    var anchored_start = False
    var anchored_end = False
    if body.startswith("^"):
        anchored_start = True
        body = _strip_prefix(body, 1)
    if body.endswith("$") and not body.endswith("\\$"):
        anchored_end = True
        body = _strip_suffix(body, 1)

    var wrapped = "(" + body + ")"
    if not anchored_start:
        wrapped = "(\\s|\\S)*" + wrapped
    if not anchored_end:
        wrapped = wrapped + "(\\s|\\S)*"
    return Regex.compile(wrapped)


# ---------------------------------------------------------------------------
# `format` attribute checks
# ---------------------------------------------------------------------------


def _is_digit(b: UInt8) -> Bool:
    return b >= 0x30 and b <= 0x39


def _is_hex(b: UInt8) -> Bool:
    if _is_digit(b):
        return True
    if b >= 0x41 and b <= 0x46:
        return True
    return b >= 0x61 and b <= 0x66


def _is_alpha(b: UInt8) -> Bool:
    if b >= 0x41 and b <= 0x5A:
        return True
    return b >= 0x61 and b <= 0x7A


def _digits_value(s: String, start: Int, count: Int) -> Int:
    """The integer spelled by `count` digits at `start`, or -1."""
    var bytes = s.as_bytes()
    if start + count > len(bytes):
        return -1
    var total = 0
    for i in range(start, start + count):
        if not _is_digit(bytes[i]):
            return -1
        total = total * 10 + Int(bytes[i] - 0x30)
    return total


def _days_in_month(year: Int, month: Int) -> Int:
    if month == 2:
        var leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
        return 29 if leap else 28
    if month == 4 or month == 6 or month == 9 or month == 11:
        return 30
    return 31


def _check_date(s: String) -> Bool:
    """RFC 3339 `full-date`, with the day checked against the month."""
    if s.byte_length() != 10:
        return False
    var bytes = s.as_bytes()
    if bytes[4] != 0x2D or bytes[7] != 0x2D:
        return False
    var year = _digits_value(s, 0, 4)
    var month = _digits_value(s, 5, 2)
    var day = _digits_value(s, 8, 2)
    if year < 0 or month < 1 or month > 12 or day < 1:
        return False
    return day <= _days_in_month(year, month)


def _check_time(s: String) -> Bool:
    """RFC 3339 `full-time`, so the offset is required.

    A leap second is accepted at `:60`, because RFC 3339 section 5.6
    allows it and rejecting it would fail a timestamp the standard
    prints as an example.
    """
    var bytes = s.as_bytes()
    var n = len(bytes)
    if n < 9:
        return False
    if bytes[2] != 0x3A or bytes[5] != 0x3A:
        return False
    var hour = _digits_value(s, 0, 2)
    var minute = _digits_value(s, 3, 2)
    var second = _digits_value(s, 6, 2)
    if hour < 0 or hour > 23:
        return False
    if minute < 0 or minute > 59:
        return False
    if second < 0 or second > 60:
        return False

    var i = 8
    if i < n and bytes[i] == 0x2E:
        i += 1
        var digits = 0
        while i < n and _is_digit(bytes[i]):
            i += 1
            digits += 1
        if digits == 0:
            return False

    if i >= n:
        return False
    if bytes[i] == 0x5A or bytes[i] == 0x7A:
        return i + 1 == n
    if bytes[i] != 0x2B and bytes[i] != 0x2D:
        return False
    if i + 6 != n:
        return False
    if bytes[i + 3] != 0x3A:
        return False
    var oh = _digits_value(s, i + 1, 2)
    var om = _digits_value(s, i + 4, 2)
    if oh < 0 or oh > 23:
        return False
    return om >= 0 and om <= 59


def _check_date_time(s: String) -> Bool:
    """RFC 3339 `date-time`: a full date, `T`, and a full time."""
    if s.byte_length() < 12:
        return False
    var bytes = s.as_bytes()
    if bytes[10] != 0x54 and bytes[10] != 0x74:
        return False
    if not _check_date(_strip_suffix(s, s.byte_length() - 10)):
        return False
    return _check_time(_strip_prefix(s, 11))


def _check_hostname(s: String) -> Bool:
    """A dot-separated RFC 1123 hostname.

    A trailing root dot is accepted: it names the same host and the
    grammar in RFC 1123 section 2.1 allows it.
    """
    var text = s
    if text.endswith(".") and text.byte_length() > 1:
        text = _strip_suffix(text, 1)
    if text == "" or text.byte_length() > 253:
        return False
    var bytes = text.as_bytes()
    var n = len(bytes)
    var label_start = 0
    var i = 0
    while i <= n:
        if i == n or bytes[i] == 0x2E:
            var length = i - label_start
            if length == 0 or length > 63:
                return False
            if bytes[label_start] == 0x2D or bytes[i - 1] == 0x2D:
                return False
            label_start = i + 1
        elif not (
            _is_alpha(bytes[i]) or _is_digit(bytes[i]) or bytes[i] == 0x2D
        ):
            return False
        i += 1
    return True


def _check_ipv4(s: String) -> Bool:
    """A dotted quad. A leading zero in an octet is rejected.

    `010.1.1.1` is read as octal by some resolvers and as decimal by
    others, so a string that two readers disagree about is not a valid
    dotted quad here.
    """
    var bytes = s.as_bytes()
    var n = len(bytes)
    var octets = 0
    var i = 0
    while i < n:
        var start = i
        while i < n and _is_digit(bytes[i]):
            i += 1
        var length = i - start
        if length == 0 or length > 3:
            return False
        if length > 1 and bytes[start] == 0x30:
            return False
        var value = _digits_value(s, start, length)
        if value > 255:
            return False
        octets += 1
        if i < n:
            if bytes[i] != 0x2E:
                return False
            i += 1
            if i == n:
                return False
    return octets == 4


def _check_ipv6(s: String) -> Bool:
    """An RFC 4291 textual IPv6 address, including the `::` form.

    A trailing dotted quad is accepted, since RFC 4291 section 2.2
    defines it and it is the usual spelling of an IPv4-mapped address.
    """
    var source = s.as_bytes()
    var source_length = len(source)
    if source_length == 0:
        return False

    var last_colon = -1
    var tail_is_ipv4 = False
    for i in range(source_length):
        if source[i] == 0x3A:
            last_colon = i
        elif source[i] == 0x2E:
            tail_is_ipv4 = True

    # A trailing IPv4 part stands in for the last two groups. The colon
    # that separates the two halves stays on the front half, so that a
    # leading `::` is still two colons once the tail is cut away.
    var groups_needed = 8
    var head = s
    if tail_is_ipv4:
        if last_colon < 0:
            return False
        if not _check_ipv4(_strip_prefix(s, last_colon + 1)):
            return False
        head = _strip_suffix(s, source_length - last_colon - 1)
        groups_needed = 6

    var bytes = head.as_bytes()
    var n = len(bytes)
    var groups = 0
    var compressions = 0
    var i = 0
    var at_field_start = True
    while i < n:
        if bytes[i] == 0x3A:
            if i + 1 < n and bytes[i + 1] == 0x3A:
                compressions += 1
                i += 2
                at_field_start = True
                continue
            if at_field_start:
                return False
            i += 1
            if i == n and not tail_is_ipv4:
                return False
            at_field_start = True
            continue
        var start = i
        while i < n and _is_hex(bytes[i]):
            i += 1
        var length = i - start
        if length == 0 or length > 4:
            return False
        groups += 1
        at_field_start = False

    if compressions > 1:
        return False
    if compressions == 1:
        return groups < groups_needed
    return groups == groups_needed


def _check_uuid(s: String) -> Bool:
    """The RFC 4122 hyphenated form, eight-four-four-four-twelve."""
    if s.byte_length() != 36:
        return False
    var bytes = s.as_bytes()
    for i in range(36):
        if i == 8 or i == 13 or i == 18 or i == 23:
            if bytes[i] != 0x2D:
                return False
        elif not _is_hex(bytes[i]):
            return False
    return True


def _check_uri(s: String) -> Bool:
    """An absolute RFC 3986 URI: a scheme, a colon, and no spaces.

    The scheme is the part a relative reference cannot have, so
    checking for it is what separates `uri` from `uri-reference`. The
    rest of the grammar is permissive enough that checking it in full
    rejects almost nothing a caller would call invalid.
    """
    var bytes = s.as_bytes()
    var n = len(bytes)
    if n == 0 or not _is_alpha(bytes[0]):
        return False
    var i = 0
    while i < n and bytes[i] != 0x3A:
        var b = bytes[i]
        if not (
            _is_alpha(b) or _is_digit(b) or b == 0x2B or b == 0x2D or b == 0x2E
        ):
            return False
        i += 1
    if i == n or i == 0:
        return False
    for j in range(n):
        if bytes[j] <= 0x20 or bytes[j] == 0x7F:
            return False
    return True


def _check_email(s: String) -> Bool:
    """One `@`, a non-empty local part, and a hostname domain.

    RFC 5322 allows quoted local parts and comments that almost no
    address uses; accepting the dot-atom form and requiring the domain
    to be a hostname rejects the mistakes callers actually make.
    """
    var bytes = s.as_bytes()
    var n = len(bytes)
    var at = -1
    for i in range(n):
        if bytes[i] == 0x40:
            if at >= 0:
                return False
            at = i
    if at <= 0 or at == n - 1:
        return False
    for i in range(at):
        var b = bytes[i]
        if b <= 0x20 or b == 0x7F or b == 0x2C or b == 0x3A:
            return False
    if bytes[0] == 0x2E or bytes[at - 1] == 0x2E:
        return False
    return _check_hostname(_strip_prefix(s, at + 1))


def _check_json_pointer(s: String) -> Bool:
    """An RFC 6901 pointer, decided by the module that implements them.

    Reimplementing the syntax rule here is how the two would drift, and
    a `format` that disagrees with `Value.at` is worse than none.
    """
    try:
        var tokens = parse_pointer(s)
        _ = len(tokens)
        return True
    except:
        return False


def _check_relative_json_pointer(s: String) -> Bool:
    """A non-negative integer, then either `#` or a JSON Pointer."""
    var bytes = s.as_bytes()
    var n = len(bytes)
    var i = 0
    while i < n and _is_digit(bytes[i]):
        i += 1
    if i == 0:
        return False
    if i > 1 and bytes[0] == 0x30:
        return False
    if i == n:
        return True
    if bytes[i] == 0x23:
        return i + 1 == n
    return _check_json_pointer(_strip_prefix(s, i))


def _check_duration(s: String) -> Bool:
    """An RFC 3339 appendix A `duration`, in its ISO 8601 spelling."""
    var bytes = s.as_bytes()
    var n = len(bytes)
    if n < 3 or bytes[0] != 0x50:
        return False
    var i = 1
    var in_time = False
    var fields = 0
    while i < n:
        if bytes[i] == 0x54:
            if in_time:
                return False
            in_time = True
            i += 1
            if i == n:
                return False
            continue
        var start = i
        while i < n and _is_digit(bytes[i]):
            i += 1
        if i == start or i == n:
            return False
        var unit = bytes[i]
        i += 1
        if in_time:
            if not (unit == 0x48 or unit == 0x4D or unit == 0x53):
                return False
        elif not (unit == 0x59 or unit == 0x4D or unit == 0x57 or unit == 0x44):
            return False
        fields += 1
    return fields > 0


def _check_format(name: String, text: String) -> Bool:
    """Whether `text` satisfies the `format` attribute `name`.

    An unrecognised name answers True. Draft 2020-12 section 7.2.3 says
    an implementation must not fail validation on a format it does not
    know, and reporting a match for one is the only reading that keeps
    the assertion mode usable.
    """
    if name == "date-time":
        return _check_date_time(text)
    if name == "date":
        return _check_date(text)
    if name == "time":
        return _check_time(text)
    if name == "duration":
        return _check_duration(text)
    if name == "email" or name == "idn-email":
        return _check_email(text)
    if name == "hostname" or name == "idn-hostname":
        return _check_hostname(text)
    if name == "ipv4":
        return _check_ipv4(text)
    if name == "ipv6":
        return _check_ipv6(text)
    if name == "uri" or name == "iri":
        return _check_uri(text)
    if name == "uuid":
        return _check_uuid(text)
    if name == "json-pointer":
        return _check_json_pointer(text)
    if name == "relative-json-pointer":
        return _check_relative_json_pointer(text)
    if name == "regex":
        try:
            var compiled = Regex.compile(text)
            _ = len(compiled.prog)
            return True
        except:
            return False
    return True


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _percent_decode(s: String) raises -> String:
    """Undo the percent-encoding a URI fragment may carry.

    A `$ref` is a URI reference, so a member named `a/b` reaches the
    fragment as `a~1b` and a member named `%` reaches it as `%25`. The
    pointer layer decodes the first; this decodes the second, so the
    two spellings resolve to the same node.
    """
    var bytes = s.as_bytes()
    var n = len(bytes)
    var out = List[UInt8](capacity=n)
    var i = 0
    while i < n:
        if bytes[i] == 0x25:
            if (
                i + 2 >= n
                or not _is_hex(bytes[i + 1])
                or not _is_hex(bytes[i + 2])
            ):
                raise Error("truncated percent-escape in $ref: " + s)
            var hi = _hex_digit(bytes[i + 1])
            var lo = _hex_digit(bytes[i + 2])
            out.append(UInt8(hi * 16 + lo))
            i += 3
        else:
            out.append(bytes[i])
            i += 1
    return String(unsafe_from_utf8=Span(out))


def _hex_digit(b: UInt8) -> Int:
    if _is_digit(b):
        return Int(b - 0x30)
    if b >= 0x61:
        return Int(b - 0x61) + 10
    return Int(b - 0x41) + 10


def _non_negative_int(value: Value, keyword: String) raises -> Int:
    """An operand the spec declares a non-negative integer."""
    if not _is_integral(value):
        raise Error("schema keyword '" + keyword + "' needs an integer")
    var n = Int(_as_float(value))
    if n < 0:
        raise Error(
            "schema keyword '" + keyword + "' needs a non-negative integer"
        )
    return n


def _string_list(value: Value, keyword: String) raises -> List[String]:
    """An operand the spec declares an array of strings."""
    if not value.is_array():
        raise Error("schema keyword '" + keyword + "' needs an array")
    var items = value.array_items()
    var out = List[String](capacity=len(items))
    for i in range(len(items)):
        if not items[i].is_string():
            raise Error(
                "schema keyword '" + keyword + "' needs an array of strings"
            )
        out.append(items[i].string_value())
    return out^


def _type_code(name: String) raises -> Int:
    if name == "null":
        return _T_NULL
    if name == "boolean":
        return _T_BOOLEAN
    if name == "object":
        return _T_OBJECT
    if name == "array":
        return _T_ARRAY
    if name == "number":
        return _T_NUMBER
    if name == "string":
        return _T_STRING
    if name == "integer":
        return _T_INTEGER
    raise Error("schema keyword 'type' does not define '" + name + "'")


def _type_code_name(code: Int) -> String:
    if code == _T_NULL:
        return "null"
    if code == _T_BOOLEAN:
        return "boolean"
    if code == _T_OBJECT:
        return "object"
    if code == _T_ARRAY:
        return "array"
    if code == _T_NUMBER:
        return "number"
    if code == _T_STRING:
        return "string"
    return "integer"


struct _Compiler(Movable):
    """Builds the node list, then resolves the references into it.

    Resolution is a second pass because a `$ref` may name a subschema
    that has not been walked yet, and a forward reference into `$defs`
    is the ordinary case rather than the exception.
    """

    var nodes: List[_Node]
    var regexes: List[Regex]
    var locations: Dict[String, Int]
    var anchors: Dict[String, Int]
    var ids: Dict[String, Int]
    var ref_nodes: List[Int]
    var ref_targets: List[String]

    def __init__(out self):
        self.nodes = List[_Node]()
        self.regexes = List[Regex]()
        self.locations = Dict[String, Int]()
        self.anchors = Dict[String, Int]()
        self.ids = Dict[String, Int]()
        self.ref_nodes = List[Int]()
        self.ref_targets = List[String]()

    def _add_pattern(mut self, pattern: String) raises -> Int:
        var index = len(self.regexes)
        self.regexes.append(_pattern_program(pattern))
        return index

    def compile_node(mut self, schema: Value, location: String) raises -> Int:
        """Walk one subschema, returning the index of its node."""
        var index = len(self.nodes)
        self.nodes.append(_Node(location))
        self.locations[location] = index

        if schema.is_bool():
            var leaf = _Node(location)
            leaf.is_bool = True
            leaf.bool_ok = schema.bool_value()
            self.nodes[index] = leaf^
            return index

        if not schema.is_object():
            raise Error(
                "a schema must be an object or a boolean, but the one at '"
                + (location if location != "" else "#")
                + "' has type "
                + _type_name(schema)
            )

        var node = _Node(location)
        var members = schema.object_items()
        for i in range(len(members)):
            var key = members[i][0]
            ref operand = members[i][1]
            var here = location + "/" + escape_token(key)
            self._apply_keyword(node, key, operand, here, index)

        if node.min_contains >= 0 or node.max_contains >= 0:
            if node.contains_node < 0:
                # Section 6.4.4 attaches both to `contains`, and without
                # it neither has an instance to count, so they are
                # annotations with nothing to annotate. Dropping them
                # silently is the shape of bug this module is fixing.
                raise Error(
                    "minContains/maxContains at '"
                    + (location if location != "" else "#")
                    + "' has no sibling 'contains'"
                )

        self.nodes[index] = node^
        return index

    def _apply_keyword(
        mut self,
        mut node: _Node,
        key: String,
        operand: Value,
        here: String,
        index: Int,
    ) raises:
        """Record one member of a schema object on its node."""
        # Core vocabulary.
        if key == "$schema" or key == "$comment" or key == "$vocabulary":
            return
        if key == "$id":
            if not operand.is_string():
                raise Error("schema keyword '$id' needs a string")
            self.ids[operand.string_value()] = index
            return
        if key == "$anchor":
            if not operand.is_string():
                raise Error("schema keyword '$anchor' needs a string")
            self.anchors[operand.string_value()] = index
            return
        if key == "$ref":
            if not operand.is_string():
                raise Error("schema keyword '$ref' needs a string")
            self.ref_nodes.append(index)
            self.ref_targets.append(operand.string_value())
            return
        if key == "$dynamicRef" or key == "$dynamicAnchor":
            raise Error(
                "'"
                + key
                + "' is not supported: its target depends on the dynamic"
                " scope of the evaluation, which cannot be resolved when"
                " the schema is compiled"
            )
        if key == "$defs" or key == "definitions":
            if not operand.is_object():
                raise Error("schema keyword '" + key + "' needs an object")
            var defs = operand.object_items()
            for i in range(len(defs)):
                _ = self.compile_node(
                    defs[i][1], here + "/" + escape_token(defs[i][0])
                )
            return

        # Validation vocabulary.
        if key == "type":
            if operand.is_string():
                node.types.append(_type_code(operand.string_value()))
            elif operand.is_array():
                var names = _string_list(operand, "type")
                if len(names) == 0:
                    raise Error("schema keyword 'type' needs a non-empty array")
                for i in range(len(names)):
                    node.types.append(_type_code(names[i]))
            else:
                raise Error(
                    "schema keyword 'type' needs a string or an array of"
                    " strings"
                )
            return
        if key == "enum":
            if not operand.is_array():
                raise Error("schema keyword 'enum' needs an array")
            node.has_enum = True
            var choices = operand.array_items()
            for i in range(len(choices)):
                node.enum_values.append(choices[i].copy())
            return
        if key == "const":
            node.has_const = True
            node.const_value = operand.copy()
            return
        if key == "multipleOf":
            if not operand.is_number():
                raise Error("schema keyword 'multipleOf' needs a number")
            if _as_float(operand) <= 0.0:
                raise Error(
                    "schema keyword 'multipleOf' needs a positive number"
                )
            node.has_multiple_of = True
            node.multiple_of = operand.copy()
            return
        if key == "maximum":
            node.has_maximum = True
            node.maximum = self._number_operand(operand, key)
            return
        if key == "exclusiveMaximum":
            node.has_exclusive_maximum = True
            node.exclusive_maximum = self._number_operand(operand, key)
            return
        if key == "minimum":
            node.has_minimum = True
            node.minimum = self._number_operand(operand, key)
            return
        if key == "exclusiveMinimum":
            node.has_exclusive_minimum = True
            node.exclusive_minimum = self._number_operand(operand, key)
            return
        if key == "maxLength":
            node.max_length = _non_negative_int(operand, key)
            return
        if key == "minLength":
            node.min_length = _non_negative_int(operand, key)
            return
        if key == "pattern":
            if not operand.is_string():
                raise Error("schema keyword 'pattern' needs a string")
            node.pattern_source = operand.string_value()
            node.pattern_index = self._add_pattern(node.pattern_source)
            return
        if key == "maxItems":
            node.max_items = _non_negative_int(operand, key)
            return
        if key == "minItems":
            node.min_items = _non_negative_int(operand, key)
            return
        if key == "uniqueItems":
            if not operand.is_bool():
                raise Error("schema keyword 'uniqueItems' needs a boolean")
            node.unique_items = operand.bool_value()
            return
        if key == "maxContains":
            node.max_contains = _non_negative_int(operand, key)
            return
        if key == "minContains":
            node.min_contains = _non_negative_int(operand, key)
            return
        if key == "maxProperties":
            node.max_properties = _non_negative_int(operand, key)
            return
        if key == "minProperties":
            node.min_properties = _non_negative_int(operand, key)
            return
        if key == "required":
            node.required = _string_list(operand, "required")
            return
        if key == "dependentRequired":
            if not operand.is_object():
                raise Error(
                    "schema keyword 'dependentRequired' needs an object"
                )
            var deps = operand.object_items()
            for i in range(len(deps)):
                node.dependent_required_keys.append(deps[i][0])
                node.dependent_required_values.append(
                    _string_list(deps[i][1], "dependentRequired")
                )
            return
        if key == "format":
            if not operand.is_string():
                raise Error("schema keyword 'format' needs a string")
            node.format_name = operand.string_value()
            return

        # Applicator vocabulary.
        if key == "allOf" or key == "anyOf" or key == "oneOf":
            if not operand.is_array():
                raise Error("schema keyword '" + key + "' needs an array")
            var branches = operand.array_items()
            for i in range(len(branches)):
                var child = self.compile_node(
                    branches[i], here + "/" + String(i)
                )
                if key == "allOf":
                    node.all_of.append(child)
                elif key == "anyOf":
                    node.any_of.append(child)
                else:
                    node.one_of.append(child)
            if key == "anyOf":
                node.has_any_of = True
            elif key == "oneOf":
                node.has_one_of = True
            return
        if key == "not":
            node.not_node = self.compile_node(operand, here)
            return
        if key == "if":
            node.if_node = self.compile_node(operand, here)
            return
        if key == "then":
            node.then_node = self.compile_node(operand, here)
            return
        if key == "else":
            node.else_node = self.compile_node(operand, here)
            return
        if key == "properties":
            if not operand.is_object():
                raise Error("schema keyword 'properties' needs an object")
            var props = operand.object_items()
            for i in range(len(props)):
                node.property_names.append(props[i][0])
                node.property_nodes.append(
                    self.compile_node(
                        props[i][1], here + "/" + escape_token(props[i][0])
                    )
                )
            return
        if key == "patternProperties":
            if not operand.is_object():
                raise Error(
                    "schema keyword 'patternProperties' needs an object"
                )
            var pats = operand.object_items()
            for i in range(len(pats)):
                node.pattern_property_source.append(pats[i][0])
                node.pattern_property_regex.append(
                    self._add_pattern(pats[i][0])
                )
                node.pattern_property_nodes.append(
                    self.compile_node(
                        pats[i][1], here + "/" + escape_token(pats[i][0])
                    )
                )
            return
        if key == "additionalProperties":
            node.additional_properties = self.compile_node(operand, here)
            return
        if key == "propertyNames":
            node.property_names_node = self.compile_node(operand, here)
            return
        if key == "unevaluatedProperties":
            node.unevaluated_properties = self.compile_node(operand, here)
            return
        if key == "dependentSchemas":
            if not operand.is_object():
                raise Error("schema keyword 'dependentSchemas' needs an object")
            var deps = operand.object_items()
            for i in range(len(deps)):
                node.dependent_schema_keys.append(deps[i][0])
                node.dependent_schema_nodes.append(
                    self.compile_node(
                        deps[i][1], here + "/" + escape_token(deps[i][0])
                    )
                )
            return
        if key == "prefixItems":
            if not operand.is_array():
                raise Error("schema keyword 'prefixItems' needs an array")
            var prefix = operand.array_items()
            for i in range(len(prefix)):
                node.prefix_items.append(
                    self.compile_node(prefix[i], here + "/" + String(i))
                )
            return
        if key == "items":
            if operand.is_array():
                # Draft-07 spelled the tuple form as an array here, and
                # 2020-12 moved it to `prefixItems`. Accepting the old
                # spelling silently would make the same document mean
                # two different things depending on the reader.
                raise Error(
                    "'items' takes a single schema in draft 2020-12; the"
                    " array form moved to 'prefixItems'"
                )
            node.items_node = self.compile_node(operand, here)
            return
        if key == "contains":
            node.contains_node = self.compile_node(operand, here)
            return
        if key == "unevaluatedItems":
            node.unevaluated_items = self.compile_node(operand, here)
            return

        # Anything else is an annotation or an unknown keyword. Section
        # 6.5 of the core specification says an unknown keyword is
        # collected as an annotation and never affects validation, so
        # it is not an error and its value is not a subschema.
        return

    def _number_operand(self, operand: Value, keyword: String) raises -> Value:
        if not operand.is_number():
            raise Error("schema keyword '" + keyword + "' needs a number")
        return operand.copy()

    def resolve_refs(mut self) raises:
        """Point every `$ref` node at the node it names."""
        for i in range(len(self.ref_nodes)):
            var target = self.ref_targets[i]
            self.nodes[self.ref_nodes[i]].ref_target = self._resolve(target)

    def _resolve(self, reference: String) raises -> Int:
        var base_index = 0
        var fragment = ""
        var hash_at = reference.find("#")
        var base = reference
        if hash_at >= 0:
            base = _strip_suffix(reference, reference.byte_length() - hash_at)
            fragment = _strip_prefix(reference, hash_at + 1)

        if base != "":
            if base not in self.ids:
                raise Error(
                    "only same-document $ref is supported, so '"
                    + reference
                    + "' cannot be resolved; declare the target with $id or"
                    " inline it"
                )
            base_index = self.ids[base]

        if fragment == "":
            return base_index

        var decoded = _percent_decode(fragment)
        if not decoded.startswith("/"):
            if decoded not in self.anchors:
                raise Error(
                    "$ref '"
                    + reference
                    + "' names an $anchor that is not declared in this document"
                )
            return self.anchors[decoded]

        var tokens = parse_pointer(decoded)
        var location = self.nodes[base_index].location + build_pointer(tokens)
        if location not in self.locations:
            raise Error(
                "$ref '"
                + reference
                + "' does not point at a subschema of this document"
            )
        return self.locations[location]


# ---------------------------------------------------------------------------
# The compiled schema
# ---------------------------------------------------------------------------


struct Schema(Movable):
    """A schema document compiled once, ready to validate many times.

    `validate(document, schema)` compiles on every call, which is the
    right shape for a one-off check and the wrong one for a loop over a
    file of records: the schema is walked, its patterns are compiled
    and its references are resolved once per document rather than once.
    Compiling separately also moves every malformed-schema error to the
    point where the schema is handed over, so a bad schema is not
    reported as a bad document.
    """

    var _nodes: List[_Node]
    var _regexes: List[Regex]
    var _root: Int
    var _assert_format: Bool

    def __init__(
        out self,
        var nodes: List[_Node],
        var regexes: List[Regex],
        root: Int,
        assert_format: Bool,
    ):
        self._nodes = nodes^
        self._regexes = regexes^
        self._root = root
        self._assert_format = assert_format

    @staticmethod
    def compile(schema: Value) raises -> Schema:
        """Compile a schema document.

        Args:
            schema: The schema, as a parsed JSON value. A boolean is a
                schema too: `true` accepts every instance and `false`
                rejects every instance.

        Returns:
            The compiled schema.

        Raises:
            Error: If the document is not a valid schema. A keyword
                with an operand of the wrong type, a `pattern` outside
                I-Regexp and a `$ref` that names nothing in the
                document are all reported here rather than turning
                into a silent pass at validation time.
        """
        return Schema.compile(schema, assert_format=False)

    @staticmethod
    def compile(schema: Value, assert_format: Bool) raises -> Schema:
        """Compile a schema document, choosing how `format` behaves.

        Args:
            schema: The schema, as a parsed JSON value.
            assert_format: True to make `format` an assertion, so a
                string that does not satisfy its format attribute
                fails. The default is the specification's own default,
                where `format` is collected as an annotation and never
                changes the verdict.

        Returns:
            The compiled schema.

        Raises:
            Error: If the document is not a valid schema.
        """
        var compiler = _Compiler()
        var root = compiler.compile_node(schema, "")
        compiler.resolve_refs()
        # A `Regex` is movable but not copyable, so the programs are
        # walked out of the compiler one at a time rather than copied.
        var regexes = List[Regex]()
        while len(compiler.regexes) > 0:
            regexes.append(compiler.regexes.pop(0))
        return Schema(compiler.nodes.copy(), regexes^, root, assert_format)

    def validate(self, document: Value) raises -> ValidationResult:
        """Validate one document, collecting every reason it failed.

        Args:
            document: The instance to check.

        Returns:
            The verdict, the errors behind it, and the `format`
            annotations collected on the way.

        Raises:
            Error: If a `$ref` cycle with no base case runs the
                evaluation past its depth budget.
        """
        var result = ValidationResult()
        var annotations = self._validate(self._root, document, "", result, 0)
        _ = annotations.items_upto
        return result^

    def is_valid(self, document: Value) raises -> Bool:
        """Whether `document` satisfies the schema.

        Args:
            document: The instance to check.

        Returns:
            True when the document validates.

        Raises:
            Error: If a `$ref` cycle with no base case runs the
                evaluation past its depth budget.
        """
        var result = self.validate(document)
        return result.valid

    # -- validation ---------------------------------------------------------

    def _validate(
        self,
        index: Int,
        value: Value,
        path: String,
        mut result: ValidationResult,
        depth: Int,
    ) raises -> _Annot:
        """Apply one node, returning what it evaluated at `path`."""
        var annot = _Annot()
        if depth > _MAX_DEPTH:
            raise Error(
                "schema evaluation nested more than "
                + String(_MAX_DEPTH)
                + " levels at '"
                + (path if path != "" else "#")
                + "'; a $ref cycle with no base case is the usual cause"
            )

        ref node = self._nodes[index]

        if node.is_bool:
            if not node.bool_ok:
                result.add_error(
                    path, node.location, "the false schema rejects everything"
                )
            return annot^

        if node.ref_target >= 0:
            # Draft 2020-12 section 8.2.3.1 applies `$ref` alongside its
            # siblings rather than replacing them, so nothing is skipped
            # here and the referenced subschema's annotations count.
            annot.merge(
                self._validate(node.ref_target, value, path, result, depth + 1)
            )

        self._check_type(node, value, path, result)
        self._check_generic(node, value, path, result)

        if value.is_number():
            self._check_number(node, value, path, result)
        if value.is_string():
            self._check_string(node, value, path, result)
        if value.is_array():
            annot.merge(self._check_array(node, value, path, result, depth))
        if value.is_object():
            annot.merge(self._check_object(node, value, path, result, depth))

        annot.merge(self._check_applicators(node, value, path, result, depth))

        # `unevaluatedProperties` and `unevaluatedItems` read the
        # annotations every other keyword of this schema produced, so
        # they run last and see the merged set.
        if value.is_object() and node.unevaluated_properties >= 0:
            annot.merge(
                self._check_unevaluated_properties(
                    node, value, path, result, depth, annot
                )
            )
        if value.is_array() and node.unevaluated_items >= 0:
            annot.merge(
                self._check_unevaluated_items(
                    node, value, path, result, depth, annot
                )
            )

        return annot^

    def _check_type(
        self,
        node: _Node,
        value: Value,
        path: String,
        mut result: ValidationResult,
    ) raises:
        if len(node.types) == 0:
            return
        for i in range(len(node.types)):
            if _matches_type(value, node.types[i]):
                return
        var wanted = String()
        for i in range(len(node.types)):
            if i > 0:
                wanted += ", "
            wanted += _type_code_name(node.types[i])
        var got = _type_name(value)
        if value.is_number() and not _is_integral(value):
            got = "number with a fractional part"
        result.add_error(
            path,
            node.location + "/type",
            "expected " + wanted + ", but the instance is a " + got,
        )

    def _check_generic(
        self,
        node: _Node,
        value: Value,
        path: String,
        mut result: ValidationResult,
    ) raises:
        if node.has_const:
            if not _deep_equal(value, node.const_value):
                result.add_error(
                    path,
                    node.location + "/const",
                    "expected the constant "
                    + String(node.const_value)
                    + ", but the instance is "
                    + String(value),
                )
        if node.has_enum:
            var found = False
            for i in range(len(node.enum_values)):
                if _deep_equal(value, node.enum_values[i]):
                    found = True
                    break
            if not found:
                result.add_error(
                    path,
                    node.location + "/enum",
                    String(value) + " is not one of the enumerated values",
                )

    def _check_number(
        self,
        node: _Node,
        value: Value,
        path: String,
        mut result: ValidationResult,
    ) raises:
        if node.has_minimum and _compare_numbers(value, node.minimum) < 0:
            result.add_error(
                path,
                node.location + "/minimum",
                String(value) + " is below the minimum " + String(node.minimum),
            )
        if node.has_maximum and _compare_numbers(value, node.maximum) > 0:
            result.add_error(
                path,
                node.location + "/maximum",
                String(value) + " is above the maximum " + String(node.maximum),
            )
        if (
            node.has_exclusive_minimum
            and _compare_numbers(value, node.exclusive_minimum) <= 0
        ):
            result.add_error(
                path,
                node.location + "/exclusiveMinimum",
                String(value)
                + " is not greater than "
                + String(node.exclusive_minimum),
            )
        if (
            node.has_exclusive_maximum
            and _compare_numbers(value, node.exclusive_maximum) >= 0
        ):
            result.add_error(
                path,
                node.location + "/exclusiveMaximum",
                String(value)
                + " is not less than "
                + String(node.exclusive_maximum),
            )
        if node.has_multiple_of and not _is_multiple_of(
            value, node.multiple_of
        ):
            result.add_error(
                path,
                node.location + "/multipleOf",
                String(value)
                + " is not a multiple of "
                + String(node.multiple_of),
            )

    def _check_string(
        self,
        node: _Node,
        value: Value,
        path: String,
        mut result: ValidationResult,
    ) raises:
        var text = value.string_value()
        if node.min_length >= 0 or node.max_length >= 0:
            var length = _code_point_count(text)
            if node.min_length >= 0 and length < node.min_length:
                result.add_error(
                    path,
                    node.location + "/minLength",
                    "the string is "
                    + String(length)
                    + " characters, below the minimum "
                    + String(node.min_length),
                )
            if node.max_length >= 0 and length > node.max_length:
                result.add_error(
                    path,
                    node.location + "/maxLength",
                    "the string is "
                    + String(length)
                    + " characters, above the maximum "
                    + String(node.max_length),
                )
        if node.pattern_index >= 0:
            if not self._regexes[node.pattern_index].full_match(text):
                result.add_error(
                    path,
                    node.location + "/pattern",
                    "the string does not match " + node.pattern_source,
                )
        if node.format_name != "":
            var matched = _check_format(node.format_name, text)
            result.format_annotations.append(
                FormatAnnotation(path, node.format_name, matched)
            )
            if self._assert_format and not matched:
                result.add_error(
                    path,
                    node.location + "/format",
                    "the string is not a valid " + node.format_name,
                )

    def _check_array(
        self,
        node: _Node,
        value: Value,
        path: String,
        mut result: ValidationResult,
        depth: Int,
    ) raises -> _Annot:
        var annot = _Annot()
        var items = value.array_items()
        var count = len(items)

        if node.min_items >= 0 and count < node.min_items:
            result.add_error(
                path,
                node.location + "/minItems",
                "the array has "
                + String(count)
                + " items, below the minimum "
                + String(node.min_items),
            )
        if node.max_items >= 0 and count > node.max_items:
            result.add_error(
                path,
                node.location + "/maxItems",
                "the array has "
                + String(count)
                + " items, above the maximum "
                + String(node.max_items),
            )
        if node.unique_items:
            for i in range(count):
                var duplicate = False
                for j in range(i + 1, count):
                    if _deep_equal(items[i], items[j]):
                        result.add_error(
                            path,
                            node.location + "/uniqueItems",
                            "items "
                            + String(i)
                            + " and "
                            + String(j)
                            + " are equal",
                        )
                        duplicate = True
                        break
                if duplicate:
                    break

        var prefix_count = len(node.prefix_items)
        if prefix_count > 0:
            var covered = min(count, prefix_count)
            for i in range(covered):
                _ = self._validate(
                    node.prefix_items[i],
                    items[i],
                    path + "/" + String(i),
                    result,
                    depth + 1,
                )
            annot.items_upto = covered

        if node.items_node >= 0:
            # The 2020-12 change from draft-07: `items` applies only to
            # the elements past whatever `prefixItems` covered.
            for i in range(prefix_count, count):
                _ = self._validate(
                    node.items_node,
                    items[i],
                    path + "/" + String(i),
                    result,
                    depth + 1,
                )
            annot.items_upto = count

        if node.contains_node >= 0:
            var matches = 0
            for i in range(count):
                var probe = ValidationResult()
                _ = self._validate(
                    node.contains_node,
                    items[i],
                    path + "/" + String(i),
                    probe,
                    depth + 1,
                )
                if probe.valid:
                    matches += 1
                    annot.extra_items.append(i)
            var wanted_min = node.min_contains if node.min_contains >= 0 else 1
            if matches < wanted_min:
                result.add_error(
                    path,
                    node.location + "/contains",
                    String(matches)
                    + " items match 'contains', below the required "
                    + String(wanted_min),
                )
            if node.max_contains >= 0 and matches > node.max_contains:
                result.add_error(
                    path,
                    node.location + "/maxContains",
                    String(matches)
                    + " items match 'contains', above the allowed "
                    + String(node.max_contains),
                )

        return annot^

    def _check_object(
        self,
        node: _Node,
        value: Value,
        path: String,
        mut result: ValidationResult,
        depth: Int,
    ) raises -> _Annot:
        var annot = _Annot()
        var members = value.object_items()
        var count = len(members)

        if node.min_properties >= 0 and count < node.min_properties:
            result.add_error(
                path,
                node.location + "/minProperties",
                "the object has "
                + String(count)
                + " properties, below the minimum "
                + String(node.min_properties),
            )
        if node.max_properties >= 0 and count > node.max_properties:
            result.add_error(
                path,
                node.location + "/maxProperties",
                "the object has "
                + String(count)
                + " properties, above the maximum "
                + String(node.max_properties),
            )

        for i in range(len(node.required)):
            if not _member_index(members, node.required[i]) >= 0:
                result.add_error(
                    path,
                    node.location + "/required",
                    "the required property '"
                    + node.required[i]
                    + "' is absent",
                )

        for i in range(len(node.dependent_required_keys)):
            var trigger = node.dependent_required_keys[i]
            if _member_index(members, trigger) < 0:
                continue
            ref needed = node.dependent_required_values[i]
            for j in range(len(needed)):
                if _member_index(members, needed[j]) < 0:
                    result.add_error(
                        path,
                        node.location
                        + "/dependentRequired/"
                        + escape_token(trigger),
                        "'"
                        + trigger
                        + "' is present, so '"
                        + needed[j]
                        + "' is required too",
                    )

        if node.property_names_node >= 0:
            for i in range(count):
                var name = Value(members[i][0])
                _ = self._validate(
                    node.property_names_node,
                    name,
                    _child_path(path, members[i][0]),
                    result,
                    depth + 1,
                )

        for i in range(len(node.property_names)):
            var at = _member_index(members, node.property_names[i])
            if at < 0:
                continue
            _ = self._validate(
                node.property_nodes[i],
                members[at][1],
                _child_path(path, node.property_names[i]),
                result,
                depth + 1,
            )
            annot.props.append(node.property_names[i])

        for i in range(count):
            var name = members[i][0]
            for j in range(len(node.pattern_property_regex)):
                if not self._regexes[node.pattern_property_regex[j]].full_match(
                    name
                ):
                    continue
                _ = self._validate(
                    node.pattern_property_nodes[j],
                    members[i][1],
                    _child_path(path, name),
                    result,
                    depth + 1,
                )
                annot.props.append(name)

        if node.additional_properties >= 0:
            # `additionalProperties` is offset against `properties` and
            # `patternProperties` together. Offsetting it against only
            # the first, as the previous version did, rejected every
            # member a `patternProperties` entry had already accepted.
            for i in range(count):
                var name = members[i][0]
                if _covered_by_properties(node, name):
                    continue
                if self._matches_any_pattern_property(node, name):
                    continue
                _ = self._validate(
                    node.additional_properties,
                    members[i][1],
                    _child_path(path, name),
                    result,
                    depth + 1,
                )
                annot.props.append(name)

        for i in range(len(node.dependent_schema_keys)):
            if _member_index(members, node.dependent_schema_keys[i]) < 0:
                continue
            annot.merge(
                self._validate(
                    node.dependent_schema_nodes[i],
                    value,
                    path,
                    result,
                    depth + 1,
                )
            )

        return annot^

    def _matches_any_pattern_property(self, node: _Node, name: String) -> Bool:
        for j in range(len(node.pattern_property_regex)):
            if self._regexes[node.pattern_property_regex[j]].full_match(name):
                return True
        return False

    def _check_applicators(
        self,
        node: _Node,
        value: Value,
        path: String,
        mut result: ValidationResult,
        depth: Int,
    ) raises -> _Annot:
        var annot = _Annot()

        for i in range(len(node.all_of)):
            annot.merge(
                self._validate(node.all_of[i], value, path, result, depth + 1)
            )

        if node.has_any_of:
            var matched = False
            var causes = ValidationResult()
            for i in range(len(node.any_of)):
                var probe = ValidationResult()
                var sub = self._validate(
                    node.any_of[i], value, path, probe, depth + 1
                )
                if probe.valid:
                    matched = True
                    annot.merge(sub)
                else:
                    _absorb(causes, probe, "anyOf branch " + String(i) + ": ")
            if not matched:
                result.add_error(
                    path,
                    node.location + "/anyOf",
                    "the instance matches none of the "
                    + String(len(node.any_of))
                    + " subschemas",
                )
                _absorb(result, causes, "")

        if node.has_one_of:
            var matches = List[Int]()
            var winner = _Annot()
            var causes = ValidationResult()
            for i in range(len(node.one_of)):
                var probe = ValidationResult()
                var sub = self._validate(
                    node.one_of[i], value, path, probe, depth + 1
                )
                if probe.valid:
                    matches.append(i)
                    if len(matches) == 1:
                        winner = sub^
                else:
                    _absorb(causes, probe, "oneOf branch " + String(i) + ": ")
            if len(matches) == 1:
                annot.merge(winner)
            elif len(matches) == 0:
                result.add_error(
                    path,
                    node.location + "/oneOf",
                    "the instance matches none of the "
                    + String(len(node.one_of))
                    + " subschemas",
                )
                _absorb(result, causes, "")
            else:
                var which = String()
                for i in range(len(matches)):
                    if i > 0:
                        which += ", "
                    which += String(matches[i])
                result.add_error(
                    path,
                    node.location + "/oneOf",
                    "the instance matches subschemas " + which + ", not one",
                )

        if node.not_node >= 0:
            var probe = ValidationResult()
            _ = self._validate(node.not_node, value, path, probe, depth + 1)
            if probe.valid:
                result.add_error(
                    path,
                    node.location + "/not",
                    "the instance matches the subschema it must not match",
                )
            # A passing `not` produces no annotations: the subschema it
            # negates failed, and a failed schema's annotations are
            # dropped.

        if node.if_node >= 0:
            var probe = ValidationResult()
            var if_annot = self._validate(
                node.if_node, value, path, probe, depth + 1
            )
            if probe.valid:
                annot.merge(if_annot)
                if node.then_node >= 0:
                    annot.merge(
                        self._validate(
                            node.then_node, value, path, result, depth + 1
                        )
                    )
            elif node.else_node >= 0:
                # The reasons `if` failed are dropped, which is what
                # makes `if` a test rather than an assertion, but the
                # reasons `else` fails are the caller's answer and go
                # straight into the reported result.
                annot.merge(
                    self._validate(
                        node.else_node, value, path, result, depth + 1
                    )
                )

        return annot^

    def _check_unevaluated_properties(
        self,
        node: _Node,
        value: Value,
        path: String,
        mut result: ValidationResult,
        depth: Int,
        seen: _Annot,
    ) raises -> _Annot:
        var annot = _Annot()
        var members = value.object_items()
        for i in range(len(members)):
            var name = members[i][0]
            if seen.covers_property(name):
                continue
            _ = self._validate(
                node.unevaluated_properties,
                members[i][1],
                _child_path(path, name),
                result,
                depth + 1,
            )
            annot.props.append(name)
        return annot^

    def _check_unevaluated_items(
        self,
        node: _Node,
        value: Value,
        path: String,
        mut result: ValidationResult,
        depth: Int,
        seen: _Annot,
    ) raises -> _Annot:
        var annot = _Annot()
        var items = value.array_items()
        for i in range(len(items)):
            if seen.covers_item(i):
                continue
            _ = self._validate(
                node.unevaluated_items,
                items[i],
                path + "/" + String(i),
                result,
                depth + 1,
            )
            annot.extra_items.append(i)
        return annot^


def _absorb(
    mut into: ValidationResult, cause: ValidationResult, prefix: String
):
    """Carry a sub-validation's errors into the reported result.

    A composition keyword that reports only "does not match any
    subschema" tells the caller nothing they did not already know. The
    reasons each branch failed are the answer to the question they are
    about to ask next, so they are kept.
    """
    for i in range(len(cause.errors)):
        into.valid = False
        into.errors.append(
            ValidationError(
                cause.errors[i].path,
                cause.errors[i].keyword_location,
                prefix + cause.errors[i].message,
            )
        )


def _member_index(members: List[Tuple[String, Value]], name: String) -> Int:
    for i in range(len(members)):
        if members[i][0] == name:
            return i
    return -1


def _covered_by_properties(node: _Node, name: String) -> Bool:
    for i in range(len(node.property_names)):
        if node.property_names[i] == name:
            return True
    return False


def _matches_type(value: Value, code: Int) -> Bool:
    if code == _T_NULL:
        return value.is_null()
    if code == _T_BOOLEAN:
        return value.is_bool()
    if code == _T_OBJECT:
        return value.is_object()
    if code == _T_ARRAY:
        return value.is_array()
    if code == _T_STRING:
        return value.is_string()
    if code == _T_NUMBER:
        return value.is_number()
    return value.is_number() and _is_integral(value)


def _is_multiple_of(value: Value, divisor: Value) -> Bool:
    """Whether dividing `value` by `divisor` gives an integer.

    Two integers are tested with an integer remainder, because the
    previous version truncated the quotient through `Int` and so
    overflowed on anything past `Int64` and lost the low bits well
    before that. Only when a fractional part is in play does the test
    move to binary64, where the quotient is compared against its own
    floor rather than against a truncated copy of itself.
    """
    if value.is_int() and divisor.is_int():
        var d = divisor.int_value()
        if d == 0:
            return False
        return value.int_value() % d == 0
    if (value.is_int() or value.is_uint()) and (
        divisor.is_int() or divisor.is_uint()
    ):
        var du = divisor.uint_value() if divisor.is_uint() else UInt64(
            divisor.int_value()
        )
        if du == 0:
            return False
        if value.is_uint():
            return value.uint_value() % du == 0
        var vi = value.int_value()
        var magnitude = UInt64(-vi) if vi < 0 else UInt64(vi)
        return magnitude % du == 0

    var num = _as_float(value)
    var den = _as_float(divisor)
    if den == 0.0:
        return False
    var quotient = num / den
    if quotient != quotient:
        return False
    return floor(quotient) == quotient


# ---------------------------------------------------------------------------
# The one-shot entry points
# ---------------------------------------------------------------------------


def validate(document: Value, schema: Value) raises -> ValidationResult:
    """Validate a JSON document against a JSON Schema (draft 2020-12).

    Every keyword of the core, applicator, validation and
    unevaluated-location vocabularies is implemented, along with
    `format` as an annotation. `$ref` resolves inside the schema
    document, through a JSON Pointer fragment or an `$anchor`.

    To check many documents against one schema, compile it once with
    `Schema.compile` instead: this entry point recompiles on every
    call.

    Args:
        document: The JSON document to validate.
        schema: The JSON Schema.

    Returns:
        The verdict, every reason behind it, and the `format`
        annotations collected on the way.

    Raises:
        Error: If the schema is not a valid schema document, or if a
            `$ref` cycle with no base case runs the evaluation past its
            depth budget.

    Example:
        var schema = loads('{"type":"object","required":["name"]}')
        var doc = loads('{"name":"Alice"}')
        var result = validate(doc, schema)
        if result:
            print("Valid!").
    """
    var compiled = Schema.compile(schema)
    return compiled.validate(document)


def is_valid(document: Value, schema: Value) raises -> Bool:
    """Check whether a document is valid against a schema.

    Args:
        document: The JSON document to validate.
        schema: The JSON Schema.

    Returns:
        True if valid, False otherwise.

    Raises:
        Error: If the schema is not a valid schema document, or if a
            `$ref` cycle with no base case runs the evaluation past its
            depth budget.
    """
    var result = validate(document, schema)
    return result.valid
