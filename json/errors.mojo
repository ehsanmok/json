from std.collections import List

# Parse error formatting.
#
# A parser error is only useful if it says where. These helpers turn a
# byte offset into a line, a column and a window of the surrounding
# text, so a failure reads as a place in the document rather than as
# an internal complaint.
#
# Everything takes a `Span[UInt8]` as well as a `String`, because the
# native parser has already borrowed the input as bytes and the error
# path should not have to rebuild a `String` to describe itself.
#
# Messages are written as the thing that is wrong, in lower case, with
# no module prefix: "unterminated string", not "Stage 2: unterminated
# string". Which pass noticed is not something a caller can act on.


def compute_line_column(
    source: Span[UInt8, _], position: Int
) -> Tuple[Int, Int]:
    """Line and column for a byte offset, both 1-indexed.

    The column counts bytes, not characters: a column past a
    multi-byte character will not line up with what an editor shows.
    Counting code points would need a second decode of the line on a
    path that only runs when something has already gone wrong, and the
    surrounding context in the message is what a reader actually uses
    to find the spot.

    Args:
        source: The document being parsed.
        position: Byte offset, 0-indexed.

    Returns:
        A (line, column) pair.
    """
    var n = len(source)
    var pos = min(position, n)

    var line = 1
    var column = 1

    for i in range(pos):
        if source[i] == UInt8(ord("\n")):
            line += 1
            column = 1
        else:
            column += 1

    return (line, column)


def compute_line_column(source: String, position: Int) -> Tuple[Int, Int]:
    """Line and column for a byte offset, both 1-indexed.

    Args:
        source: The document being parsed.
        position: Byte offset, 0-indexed.

    Returns:
        A (line, column) pair.
    """
    return compute_line_column(source.as_bytes(), position)


def format_error_context(
    source: Span[UInt8, _], position: Int, context_chars: Int = 20
) -> String:
    """A window of the document around `position`, for the message.

    Bytes are copied through rather than passed to `chr`, which would
    re-encode every byte at or above 0x80 as its own code point and
    turn any non-ASCII text in the window into mojibake. The window
    can start or end mid-character, so the result is assembled as
    bytes and only then called a string.

    Args:
        source: The document being parsed.
        position: Byte offset of the error.
        context_chars: How many bytes to show on each side.

    Returns:
        The window, with control characters escaped.
    """
    var n = len(source)
    var pos = min(position, n)

    var start = max(0, pos - context_chars)
    var end = min(n, pos + context_chars)

    var out = List[UInt8](capacity=end - start + 8)
    if start > 0:
        out.append(UInt8(ord(".")))
        out.append(UInt8(ord(".")))
        out.append(UInt8(ord(".")))

    for i in range(start, end):
        var c = source[i]
        if c == UInt8(ord("\n")):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("n")))
        elif c == UInt8(ord("\r")):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("r")))
        elif c == UInt8(ord("\t")):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("t")))
        else:
            out.append(c)

    if end < n:
        out.append(UInt8(ord(".")))
        out.append(UInt8(ord(".")))
        out.append(UInt8(ord(".")))

    return String(unsafe_from_utf8=out^)


def format_error_context(
    source: String, position: Int, context_chars: Int = 20
) -> String:
    """A window of the document around `position`, for the message.

    Args:
        source: The document being parsed.
        position: Byte offset of the error.
        context_chars: How many bytes to show on each side.

    Returns:
        The window, with control characters escaped.
    """
    return format_error_context(source.as_bytes(), position, context_chars)


def json_parse_error(
    message: String, source: Span[UInt8, _], position: Int
) -> String:
    """Build the full parse-error text for a byte offset.

    Args:
        message: What is wrong, lower case, no module prefix.
        source: The document being parsed.
        position: Byte offset of the error.

    Returns:
        The message with its line, column, offset and context.

    Example:
        `JSON parse error at line 3, column 15 (offset 42): expected ':' after object key`
        `  Near: ..."key" value}...`
    """
    var line_col = compute_line_column(source, position)
    return (
        "JSON parse error at line "
        + String(line_col[0])
        + ", column "
        + String(line_col[1])
        + " (offset "
        + String(position)
        + "): "
        + message
        + "\n  Near: "
        + format_error_context(source, position)
    )


def json_parse_error(message: String, source: String, position: Int) -> String:
    """Build the full parse-error text for a byte offset.

    Args:
        message: What is wrong, lower case, no module prefix.
        source: The document being parsed.
        position: Byte offset of the error.

    Returns:
        The message with its line, column, offset and context.
    """
    return json_parse_error(message, source.as_bytes(), position)


def parse_error(
    message: String, source: Span[UInt8, _], position: Int
) -> Error:
    """A located parse error, ready to raise.

    Returns the error rather than raising it so that `raise` stays at
    the call site. A helper that raised on the caller's behalf would
    read fine but hide the control flow: the compiler would treat the
    call as one that can return, and every variable assigned on the
    surviving branch would look possibly-uninitialized.

    Args:
        message: What is wrong, lower case, no module prefix.
        source: The document being parsed.
        position: Byte offset of the error.

    Returns:
        The error to raise.
    """
    return Error(json_parse_error(message, source, position))


def find_error_position(source: String) -> Int:
    """Guess where a parse failed, for a backend that does not say.

    The native parser reports its own offsets, so this is only for the
    simdjson backend, whose error codes carry no position.

    Args:
        source: The source JSON string.

    Returns:
        Estimated error position (byte offset).
    """
    var source_bytes = source.as_bytes()
    var n = len(source_bytes)
    var depth = 0
    var in_string = False
    var escaped = False
    var last_structural = 0

    for i in range(n):
        var c = source_bytes[i]

        if escaped:
            escaped = False
            continue

        if c == UInt8(ord("\\")) and in_string:
            escaped = True
            continue

        if c == UInt8(ord('"')):
            in_string = not in_string
            last_structural = i
            continue

        if in_string:
            continue

        # Structural characters
        if c == UInt8(ord("{")) or c == UInt8(ord("[")):
            depth += 1
            last_structural = i
        elif c == UInt8(ord("}")) or c == UInt8(ord("]")):
            depth -= 1
            if depth < 0:
                return i  # Unmatched closing bracket
            last_structural = i
        elif c == UInt8(ord(":")) or c == UInt8(ord(",")):
            last_structural = i

    # If we end with unclosed brackets, return the last structural position
    if depth > 0 or in_string:
        return last_structural

    return n  # End of source
