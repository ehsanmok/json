# json - Parser and Serializer configuration

from std.collections import List


struct ParserConfig:
    """Configuration options for JSON parsing.

    Example:
        var config = ParserConfig(max_depth=10, allow_comments=True)
        var data = loads(json_str, config).
    """

    var max_depth: Int
    """Maximum nesting depth allowed. 0 = unlimited (default)."""

    var allow_comments: Bool
    """Allow JavaScript-style comments (// and /* */). Default: False."""

    var allow_trailing_comma: Bool
    """Allow trailing commas in arrays and objects. Default: False."""

    var ijson: Bool
    """Enforce I-JSON (RFC 7493). Default: False.

    I-JSON is RFC 8259 narrowed to what implementations agree on. Two
    of its rules can be checked against a document: an object may not
    name a member twice (section 2.1), and a string may not hold an
    unpaired surrogate (section 2.3). Both are accepted by a plain
    RFC 8259 parser, correctly, so they are opt-in here."""

    def __init__(
        out self,
        max_depth: Int = 0,
        allow_comments: Bool = False,
        allow_trailing_comma: Bool = False,
        ijson: Bool = False,
    ):
        """Create parser configuration.

        Args:
            max_depth: Maximum nesting depth (0 = unlimited).
            allow_comments: Allow // and /* */ comments.
            allow_trailing_comma: Allow trailing commas.
            ijson: Enforce the checkable rules of RFC 7493.
        """
        self.max_depth = max_depth
        self.allow_comments = allow_comments
        self.allow_trailing_comma = allow_trailing_comma
        self.ijson = ijson

    @staticmethod
    def default() -> Self:
        """Create default (strict) parser configuration."""
        return Self()

    @staticmethod
    def lenient() -> Self:
        """Create lenient parser configuration allowing common extensions."""
        return Self(
            max_depth=0,
            allow_comments=True,
            allow_trailing_comma=True,
        )

    @staticmethod
    def interoperable() -> Self:
        """Create an I-JSON (RFC 7493) parser configuration.

        Stricter than the default: the extensions stay off and the two
        checkable I-JSON rules are enforced.
        """
        return Self(ijson=True)


struct SerializerConfig:
    """Configuration options for JSON serialization.

    Example:
        var config = SerializerConfig(sort_keys=True, indent="  ")
        var json = dumps(value, config).
    """

    var indent: String
    """Indentation string. Empty = compact output (default)."""

    var sort_keys: Bool
    """Sort object keys alphabetically. Default: False."""

    var escape_unicode: Bool
    """Escape non-ASCII characters as \\uXXXX. Default: False."""

    var escape_forward_slash: Bool
    """Escape forward slashes as \\/. Default: False (for HTML safety)."""

    def __init__(
        out self,
        indent: String = "",
        sort_keys: Bool = False,
        escape_unicode: Bool = False,
        escape_forward_slash: Bool = False,
    ):
        """Create serializer configuration.

        Args:
            indent: Indentation string (empty = compact).
            sort_keys: Sort object keys alphabetically.
            escape_unicode: Escape non-ASCII as \\uXXXX.
            escape_forward_slash: Escape / as \\/ (for HTML embedding).
        """
        self.indent = indent
        self.sort_keys = sort_keys
        self.escape_unicode = escape_unicode
        self.escape_forward_slash = escape_forward_slash

    @staticmethod
    def default() -> Self:
        """Create default serializer configuration (compact output)."""
        return Self()

    @staticmethod
    def pretty(indent: String = "  ") -> Self:
        """Create pretty-print configuration with given indent."""
        return Self(indent=indent)


# Preprocessing functions for parser config


def preprocess_json(json: String, config: ParserConfig) raises -> String:
    """Preprocess JSON according to config options.

    This handles:
    - Stripping comments if allow_comments is True.
    - Removing trailing commas if allow_trailing_comma is True.
    - Checking max_depth if specified.

    Args:
        json: Input JSON string.
        config: Parser configuration.

    Returns:
        Preprocessed JSON string.
    """
    var result = json

    if config.allow_comments:
        result = _strip_comments(result)

    if config.allow_trailing_comma:
        result = _remove_trailing_commas(result)

    if config.max_depth > 0:
        _check_depth(result, config.max_depth)

    return result^


comptime _QUOTE = UInt8(0x22)
comptime _BACKSLASH = UInt8(0x5C)
comptime _SLASH = UInt8(0x2F)
comptime _STAR = UInt8(0x2A)
comptime _COMMA = UInt8(0x2C)
comptime _NEWLINE = UInt8(0x0A)
comptime _CLOSE_BRACKET = UInt8(0x5D)
comptime _CLOSE_BRACE = UInt8(0x7D)


@always_inline
def _is_space(c: UInt8) -> Bool:
    return c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D


def _strip_comments(json: String) -> String:
    """Remove JavaScript-style comments from JSON.

    Handles:
    - // single-line comments.
    - /* multi-line comments */.

    Bytes are copied as bytes. Copying them through `chr` reads a byte
    above 0x7F as a code point and writes it back as two, so any
    document with non-ASCII text came out of here corrupted before the
    parser ever saw it.
    """
    var b = json.as_bytes()
    var n = len(b)
    var out = List[UInt8](capacity=n)
    var i = 0
    var in_string = False
    var escaped = False

    while i < n:
        var c = b[i]

        if escaped:
            escaped = False
            out.append(c)
            i += 1
            continue

        if c == _BACKSLASH and in_string:
            escaped = True
            out.append(c)
            i += 1
            continue

        if c == _QUOTE:
            in_string = not in_string
            out.append(c)
            i += 1
            continue

        if not in_string and c == _SLASH and i + 1 < n:
            var next_c = b[i + 1]

            if next_c == _SLASH:
                i += 2
                while i < n and b[i] != _NEWLINE:
                    i += 1
                continue

            if next_c == _STAR:
                i += 2
                while i + 1 < n:
                    if b[i] == _STAR and b[i + 1] == _SLASH:
                        i += 2
                        break
                    i += 1
                continue

        out.append(c)
        i += 1

    return String(unsafe_from_utf8=Span(out))


def _remove_trailing_commas(json: String) -> String:
    """Remove trailing commas from arrays and objects.

    Two passes: find the commas to drop, then copy everything else.
    The previous version rebuilt the whole accumulated result string
    every time it found one, which made a document with many trailing
    commas quadratic, and it also copied bytes through `chr`.
    """
    var b = json.as_bytes()
    var n = len(b)
    var drops = List[Int]()
    var in_string = False
    var escaped = False
    var last_comma = -1

    for i in range(n):
        var c = b[i]
        if escaped:
            escaped = False
            continue
        if in_string:
            if c == _BACKSLASH:
                escaped = True
            elif c == _QUOTE:
                in_string = False
                last_comma = -1
            continue
        if c == _QUOTE:
            in_string = True
            last_comma = -1
        elif c == _COMMA:
            last_comma = i
        elif _is_space(c):
            # Whitespace between the comma and the bracket does not end
            # the run, which is the whole reason the position is kept.
            continue
        else:
            if (c == _CLOSE_BRACKET or c == _CLOSE_BRACE) and last_comma >= 0:
                drops.append(last_comma)
            last_comma = -1

    if len(drops) == 0:
        return String(unsafe_from_utf8=b)

    var out = List[UInt8](capacity=n)
    var d = 0
    for i in range(n):
        if d < len(drops) and drops[d] == i:
            d += 1
            continue
        out.append(b[i])
    return String(unsafe_from_utf8=Span(out))


def _check_depth(json: String, max_depth: Int) raises:
    """Check that JSON doesn't exceed max nesting depth."""
    var json_bytes = json.as_bytes()
    var n = len(json_bytes)
    var depth = 0
    var in_string = False
    var escaped = False

    for i in range(n):
        var c = json_bytes[i]

        if escaped:
            escaped = False
            continue

        if c == UInt8(ord("\\")) and in_string:
            escaped = True
            continue

        if c == UInt8(ord('"')):
            in_string = not in_string
            continue

        if in_string:
            continue

        if c == UInt8(ord("{")) or c == UInt8(ord("[")):
            depth += 1
            if depth > max_depth:
                raise Error(
                    "JSON exceeds maximum depth of " + String(max_depth)
                )
        elif c == UInt8(ord("}")) or c == UInt8(ord("]")):
            depth -= 1
