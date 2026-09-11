# json - Raw-string helpers for arrays/objects.
#
# These functions parse and surgically rewrite JSON arrays/objects stored
# as raw substrings on `Value`. They are PURE STRING operations -- they
# do NOT depend on the `Value` type, which keeps the import graph
# acyclic across the `value/` package.
#
# Mutation surgery (`_update_*`, `_add_*`, `_append_*`) is not
# implemented here -- mutation uses copy-on-write through `OwnedValue`
# in `value/owned.mojo`. This module is the read-side `_extract_*`
# helpers used for `raw_json()` round-trip and for `LazyValue`.

from std.collections import List


# ---------------------------------------------------------------------------
# JSON string escaping -- the single canonical implementation
# ---------------------------------------------------------------------------
#
# This lives here, in the dependency-free module, because all three
# serialization paths need it: `serialize.mojo` (scalars), `value.mojo`
# (tape-backed containers) and `owned.mojo` (owned-tree containers).
#
# It used to be four near-duplicate copies, and they had drifted apart:
# the owned-tree copy escaped only quote, backslash, 0x0A, 0x0D and 0x09
# and emitted every other control byte raw. RFC 8259 s7 requires the whole
# U+0000..U+001F range, so a control character inside a hand-built object
# produced invalid JSON, while the same value read from a tape produced
# correct output. Keeping one implementation is what stops that class of
# divergence.


comptime _HEX_DIGITS = "0123456789abcdef".as_bytes()


def escape_json_bytes(mut out: List[UInt8], s: String):
    """Append `s` to `out` as a quoted, escaped JSON string literal.

    Appends into a caller-owned buffer so a container can escape many
    keys and values into one allocation.

    Escapes per RFC 8259 s7: quote and backslash always; 0x08, 0x0C,
    0x0A, 0x0D and 0x09 as their two-character short forms; every other
    byte below 0x20 as a six-character "u00" hex escape.
    Bytes >= 0x20 pass through unchanged -- that includes 0x7F (DEL),
    which the RFC does *not* require escaping, and all UTF-8
    continuation bytes, so multi-byte characters survive untouched.
    """
    out.append(UInt8(ord('"')))
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = b[i]
        if c == UInt8(ord('"')):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord('"')))
        elif c == UInt8(ord("\\")):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("\\")))
        elif c == UInt8(0x0A):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("n")))
        elif c == UInt8(0x0D):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("r")))
        elif c == UInt8(0x09):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("t")))
        elif c == UInt8(0x08):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("b")))
        elif c == UInt8(0x0C):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("f")))
        elif c < UInt8(0x20):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("u")))
            out.append(UInt8(ord("0")))
            out.append(UInt8(ord("0")))
            out.append(_HEX_DIGITS[Int(c) >> 4])
            out.append(_HEX_DIGITS[Int(c) & 0xF])
        else:
            out.append(c)
    out.append(UInt8(ord('"')))


def escape_json_string(s: String) -> String:
    """`s` as a quoted, escaped JSON string literal.

    Builds the bytes once and converts once, rather than the previous
    `out += chr(Int(c))` per byte -- which constructed a heap `String`
    for every character in the input.
    """
    var out = List[UInt8](capacity=s.byte_length() + 2)
    escape_json_bytes(out, s)
    return String(unsafe_from_utf8=out)


# ---------------------------------------------------------------------------
# Read-side helpers (used by Value reads, LazyValue, JSON Pointer)
# ---------------------------------------------------------------------------


def _extract_field_value(raw: String, key: String) raises -> String:
    """Extract a field's value from raw JSON object string.

    Args:
        raw: Raw JSON object string (e.g., '{"a": 1, "b": "hello"}').
        key: Field name to extract.

    Returns:
        The raw JSON value as a string (e.g., '1' or '"hello"').
    """
    var raw_bytes = raw.as_bytes()
    var in_string = False
    var i = 0
    var n = len(raw_bytes)

    while i < n and (
        raw_bytes[i] == UInt8(ord("{"))
        or raw_bytes[i] == UInt8(ord(" "))
        or raw_bytes[i] == UInt8(ord("\t"))
        or raw_bytes[i] == UInt8(ord("\n"))
    ):
        i += 1

    while i < n:
        while i < n and (
            raw_bytes[i] == UInt8(ord(" "))
            or raw_bytes[i] == UInt8(ord("\t"))
            or raw_bytes[i] == UInt8(ord("\n"))
        ):
            i += 1

        if i >= n:
            break

        if raw_bytes[i] == UInt8(ord('"')) and not in_string:
            i += 1
            var key_start = i

            while i < n and raw_bytes[i] != UInt8(ord('"')):
                if raw_bytes[i] == UInt8(ord("\\")):
                    i += 2
                else:
                    i += 1

            var found_key = String(unsafe_from_utf8=raw.as_bytes()[key_start:i])
            i += 1

            while i < n and (
                raw_bytes[i] == UInt8(ord(" "))
                or raw_bytes[i] == UInt8(ord("\t"))
                or raw_bytes[i] == UInt8(ord("\n"))
                or raw_bytes[i] == UInt8(ord(":"))
            ):
                i += 1

            if found_key == key:
                return _extract_json_value(raw, i)
            else:
                _ = _extract_json_value(raw, i)
                while (
                    i < n
                    and raw_bytes[i] != UInt8(ord(","))
                    and raw_bytes[i] != UInt8(ord("}"))
                ):
                    i += 1
                if i < n and raw_bytes[i] == UInt8(ord(",")):
                    i += 1
        else:
            i += 1

    raise Error("Key not found in JSON object")


def _extract_json_value(raw: String, start: Int) raises -> String:
    """Extract a single JSON value starting at position start."""
    var raw_bytes = raw.as_bytes()
    var i = start
    var n = len(raw_bytes)

    while i < n and (
        raw_bytes[i] == UInt8(ord(" "))
        or raw_bytes[i] == UInt8(ord("\t"))
        or raw_bytes[i] == UInt8(ord("\n"))
    ):
        i += 1

    if i >= n:
        raise Error("Unexpected end of JSON")

    var first_char = raw_bytes[i]

    if first_char == UInt8(ord('"')):
        var value_start = i
        i += 1
        while i < n:
            if raw_bytes[i] == UInt8(ord("\\")):
                i += 2
                continue
            elif raw_bytes[i] == UInt8(ord('"')):
                return String(
                    String(unsafe_from_utf8=raw.as_bytes()[value_start : i + 1])
                )
            else:
                i += 1
        raise Error("Unterminated string")

    elif first_char == UInt8(ord("{")) or first_char == UInt8(ord("[")):
        var close_char = UInt8(ord("}")) if first_char == UInt8(
            ord("{")
        ) else UInt8(ord("]"))
        var depth = 1
        var value_start = i
        i += 1
        var in_string = False

        while i < n and depth > 0:
            if raw_bytes[i] == UInt8(ord("\\")) and in_string:
                i += 2
                continue
            elif raw_bytes[i] == UInt8(ord('"')):
                in_string = not in_string
            elif not in_string:
                if raw_bytes[i] == first_char:
                    depth += 1
                elif raw_bytes[i] == close_char:
                    depth -= 1
            i += 1

        return String(String(unsafe_from_utf8=raw.as_bytes()[value_start:i]))

    else:
        var value_start = i
        while (
            i < n
            and raw_bytes[i] != UInt8(ord(","))
            and raw_bytes[i] != UInt8(ord("}"))
            and raw_bytes[i] != UInt8(ord("]"))
            and raw_bytes[i] != UInt8(ord(" "))
            and raw_bytes[i] != UInt8(ord("\t"))
            and raw_bytes[i] != UInt8(ord("\n"))
        ):
            i += 1
        return String(String(unsafe_from_utf8=raw.as_bytes()[value_start:i]))


def _extract_array_element(raw: String, index: Int) raises -> String:
    """Extract an array element by index from raw JSON array string."""
    var raw_bytes = raw.as_bytes()
    var n = len(raw_bytes)
    var i = 0
    var current_index = 0

    while i < n and (
        raw_bytes[i] == UInt8(ord(" "))
        or raw_bytes[i] == UInt8(ord("\t"))
        or raw_bytes[i] == UInt8(ord("\n"))
        or raw_bytes[i] == UInt8(ord("\r"))
    ):
        i += 1

    if i >= n or raw_bytes[i] != UInt8(ord("[")):
        raise Error("Invalid JSON array")
    i += 1

    while i < n:
        while i < n and (
            raw_bytes[i] == UInt8(ord(" "))
            or raw_bytes[i] == UInt8(ord("\t"))
            or raw_bytes[i] == UInt8(ord("\n"))
            or raw_bytes[i] == UInt8(ord("\r"))
        ):
            i += 1

        if i >= n:
            break

        if raw_bytes[i] == UInt8(ord("]")):
            break

        if current_index == index:
            return _extract_json_value(raw, i)

        _ = _extract_json_value(raw, i)

        var element_depth = 0
        var in_string = False
        var escaped = False

        while i < n:
            var c = raw_bytes[i]
            if escaped:
                escaped = False
                i += 1
                continue
            if c == UInt8(ord("\\")) and in_string:
                escaped = True
                i += 1
                continue
            if c == UInt8(ord('"')):
                in_string = not in_string
                i += 1
                continue
            if in_string:
                i += 1
                continue
            if c == UInt8(ord("[")) or c == UInt8(ord("{")):
                element_depth += 1
            elif c == UInt8(ord("]")) or c == UInt8(ord("}")):
                if element_depth > 0:
                    element_depth -= 1
                else:
                    break
            elif c == UInt8(ord(",")) and element_depth == 0:
                i += 1
                current_index += 1
                break
            i += 1

        if i >= n or raw_bytes[i] == UInt8(ord("]")):
            break

    raise Error("Array index out of bounds: " + String(index))


def _count_array_elements(raw: String) -> Int:
    """Count elements in a JSON array."""
    var raw_bytes = raw.as_bytes()
    var n = len(raw_bytes)
    var count = 0
    var depth = 0
    var in_string = False
    var escaped = False

    for i in range(n):
        var c = raw_bytes[i]
        if escaped:
            escaped = False
            continue
        if c == UInt8(ord("\\")):
            escaped = True
            continue
        if c == UInt8(ord('"')):
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == UInt8(ord("[")) or c == UInt8(ord("{")):
            depth += 1
        elif c == UInt8(ord("]")) or c == UInt8(ord("}")):
            depth -= 1
        elif c == UInt8(ord(",")) and depth == 1:
            count += 1

    # `count` is the number of top-level commas. The element count is
    # `commas + 1` unless the array body is empty / whitespace-only.
    # Find the byte immediately after `[` and the byte immediately
    # before the matching closing `]`, then look for any non-ws byte
    # in between. (Nested brackets, strings, and escapes all count as
    # content; a previous version of this routine missed those cases
    # and silently returned 0 for arrays like `[[]]` or `[1]`.)
    var has_content = False
    var open_at = -1
    for i in range(n):
        if raw_bytes[i] == UInt8(ord("[")):
            open_at = i
            break
    var close_at = -1
    for i in range(n - 1, -1, -1):
        if raw_bytes[i] == UInt8(ord("]")):
            close_at = i
            break
    if open_at >= 0 and close_at > open_at:
        for j in range(open_at + 1, close_at):
            var c = raw_bytes[j]
            if (
                c != UInt8(ord(" "))
                and c != UInt8(ord("\t"))
                and c != UInt8(ord("\n"))
                and c != UInt8(ord("\r"))
            ):
                has_content = True
                break

    if has_content:
        count += 1

    return count


def _extract_object_keys(raw: String) -> List[String]:
    """Extract all keys from a JSON object."""
    var keys = List[String]()
    var raw_bytes = raw.as_bytes()
    var n = len(raw_bytes)
    var depth = 0
    var in_string = False
    var escaped = False
    var key_start = -1
    var expect_key = True

    for i in range(n):
        var c = raw_bytes[i]
        if escaped:
            escaped = False
            continue
        if c == UInt8(ord("\\")):
            escaped = True
            continue
        if c == UInt8(ord('"')):
            if not in_string:
                in_string = True
                if depth == 1 and expect_key:
                    key_start = i + 1
            else:
                in_string = False
                if key_start >= 0 and depth == 1:
                    _ = i - key_start
                    keys.append(
                        String(
                            String(unsafe_from_utf8=raw.as_bytes()[key_start:i])
                        )
                    )
                    key_start = -1
            continue
        if in_string:
            continue
        if c == UInt8(ord("{")) or c == UInt8(ord("[")):
            depth += 1
        elif c == UInt8(ord("}")) or c == UInt8(ord("]")):
            depth -= 1
        elif c == UInt8(ord(":")) and depth == 1:
            expect_key = False
        elif c == UInt8(ord(",")) and depth == 1:
            expect_key = True

    return keys^


# ---------------------------------------------------------------------------
# Mutation flows go through `OwnedValue` and `_set_at_pointer` in
# `value/owned.mojo`; this module is read-only string surgery only.
# ---------------------------------------------------------------------------


def _parse_json_pointer(pointer: String) raises -> List[String]:
    """Parse a JSON Pointer string into tokens.

    Handles RFC 6901 escape sequences:
        ~0 -> ~.
        ~1 -> /.

    Lives in `raw_ops.mojo` because it has no dependency on `Value` (it
    only inspects the pointer string and returns tokens).
    """
    var tokens = List[String]()
    var pointer_bytes = pointer.as_bytes()
    var n = len(pointer_bytes)
    var i = 1  # Skip leading /

    while i < n:
        var token = String()
        while i < n and pointer_bytes[i] != UInt8(ord("/")):
            if pointer_bytes[i] == UInt8(ord("~")):
                if i + 1 < n:
                    if pointer_bytes[i + 1] == UInt8(ord("0")):
                        token += "~"
                        i += 2
                        continue
                    elif pointer_bytes[i + 1] == UInt8(ord("1")):
                        token += "/"
                        i += 2
                        continue
                raise Error("Invalid escape sequence in JSON Pointer")
            token += chr(Int(pointer_bytes[i]))
            i += 1
        tokens.append(token^)
        i += 1  # Skip /

    return tokens^
