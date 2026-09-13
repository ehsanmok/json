# I-JSON (RFC 7493).
#
# I-JSON is RFC 8259 narrowed to the subset that different
# implementations agree on. Two of its rules are checkable and this
# module checks them:
#
#   Section 2.1: an object's member names must be unique. RFC 8259
#   merely says behaviour is unpredictable when they are not, so a
#   plain parser is right to accept `{"a":1,"a":2}`; an I-JSON parser
#   is not.
#
#   Section 2.3: a string must not contain an unpaired surrogate. RFC
#   8259 only discourages it, and this library follows section 8.2 by
#   turning an unpaired escape into U+FFFD, which is lossy: by the
#   time a document is parsed, `\uD800` and a literal U+FFFD look the
#   same. The check therefore reads the source text rather than the
#   parsed value.
#
# The rest of RFC 7493 is either already enforced (UTF-8 encoding, the
# RFC 8259 grammar) or stated at SHOULD level about protocol design
# rather than about a document, and a parser cannot check it. Section
# 2.2's advice about integers outside the range a `Float64` names
# exactly is one of those: this library keeps such integers exact, so
# rejecting them would lose information rather than protect it.

from .value import Value

comptime _QUOTE = UInt8(0x22)
comptime _BACKSLASH = UInt8(0x5C)
comptime _LOWER_U = UInt8(0x75)


def check_ijson(text: String, value: Value) raises:
    """Whether `text` and its parse satisfy the checkable I-JSON rules.

    Args:
        text: The document as it arrived, needed for the surrogate
            rule because parsing does not preserve the distinction.
        value: The same document parsed, needed for the uniqueness
            rule because that one is about structure.

    Raises:
        Error: On the first violation, naming the rule and the place.
    """
    check_no_unpaired_surrogates(text)
    check_unique_member_names(value)


def check_no_unpaired_surrogates(text: String) raises:
    """RFC 7493 section 2.3: no unpaired surrogate in any string."""
    var b = text.as_bytes()
    var n = len(b)
    var i = 0
    var in_string = False

    while i < n:
        var c = b[i]
        if not in_string:
            if c == _QUOTE:
                in_string = True
            i += 1
            continue

        if c == _QUOTE:
            in_string = False
            i += 1
            continue

        if c != _BACKSLASH or i + 1 >= n:
            i += 1
            continue

        if b[i + 1] != _LOWER_U:
            # Any other escape is two bytes, and skipping both is what
            # keeps `\\"` from being read as the end of the string.
            i += 2
            continue

        if i + 5 >= n:
            i += 2
            continue
        var first = _hex4(b, i + 2)
        if first < 0:
            i += 2
            continue

        if first >= 0xDC00 and first <= 0xDFFF:
            raise Error(
                "I-JSON (RFC 7493 section 2.3): a low surrogate escape"
                " stands alone at byte "
                + String(i)
            )
        if first < 0xD800 or first > 0xDBFF:
            i += 6
            continue

        # A high surrogate has to be followed by a low one, spelled as
        # its own escape. Anything else leaves it unpaired.
        var paired = False
        if i + 11 < n and b[i + 6] == _BACKSLASH and b[i + 7] == _LOWER_U:
            var second = _hex4(b, i + 8)
            if second >= 0xDC00 and second <= 0xDFFF:
                paired = True
        if not paired:
            raise Error(
                "I-JSON (RFC 7493 section 2.3): a high surrogate escape is"
                " unpaired at byte "
                + String(i)
            )
        i += 12


def check_unique_member_names(value: Value) raises:
    """RFC 7493 section 2.1: no object names a member twice."""
    if value.is_object():
        var members = value.object_items()
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if members[i][0] == members[j][0]:
                    raise Error(
                        "I-JSON (RFC 7493 section 2.1): the member name '"
                        + members[i][0]
                        + "' appears more than once in the same object"
                    )
        for i in range(len(members)):
            check_unique_member_names(members[i][1])
        return

    if value.is_array():
        var items = value.array_items()
        for i in range(len(items)):
            check_unique_member_names(items[i])


def _hex4(b: Span[UInt8, _], at: Int) -> Int:
    """Four hex digits read as a number, or -1 if they are not four."""
    var total = 0
    for k in range(4):
        var c = b[at + k]
        var digit: Int
        if c >= 0x30 and c <= 0x39:
            digit = Int(c) - 0x30
        elif c >= 0x61 and c <= 0x66:
            digit = Int(c) - 0x61 + 10
        elif c >= 0x41 and c <= 0x46:
            digit = Int(c) - 0x41 + 10
        else:
            return -1
        total = total * 16 + digit
    return total
