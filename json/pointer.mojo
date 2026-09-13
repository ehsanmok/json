# JSON Pointer (RFC 6901).
#
# There used to be three of these: one in `value/raw_ops.mojo` for
# `Value.at`, one in `lazy.mojo` for `LazyValue.get`, and one in
# `patch.mojo` for the operation paths. They disagreed, and a patch
# operation could be validated by one and executed by another. The
# pointer `"/"` names the member whose key is the empty string, and
# two of the three returned the whole document for it instead; `"/a/"`
# lost its trailing empty token the same way; one accepted `~2` as a
# literal tilde where the others rejected it.
#
# This module is the single answer. It depends on nothing else in the
# package so every layer can reach it.

comptime _SLASH = UInt8(0x2F)
comptime _TILDE = UInt8(0x7E)
comptime _ZERO = UInt8(0x30)
comptime _ONE = UInt8(0x31)
comptime _NINE = UInt8(0x39)
comptime _MINUS = UInt8(0x2D)


def parse_pointer(pointer: String) raises -> List[String]:
    """The reference tokens of a JSON Pointer, unescaped.

    Args:
        pointer: A JSON Pointer. The empty string names the whole
            document and yields no tokens; anything else must begin
            with a solidus.

    Returns:
        One entry per reference token, with `~1` decoded to `/` and
        `~0` to `~`.

    Raises:
        Error: If the pointer does not begin with a solidus, or a `~`
            is followed by anything other than `0` or `1`.
    """
    var tokens = List[String]()
    if pointer == "":
        return tokens^

    var bytes = pointer.as_bytes()
    var n = len(bytes)
    if bytes[0] != _SLASH:
        raise Error("JSON Pointer must be empty or start with '/': " + pointer)

    # Every solidus introduces a token, so a pointer of `n` slashes has
    # `n` tokens and the loop must run once more after the last one.
    # Reading this as "split on '/' and drop the first field" is what
    # the older copies got wrong: it silently dropped a trailing empty
    # token, and `"/"` came out as no tokens at all rather than one
    # empty one.
    var i = 1
    while True:
        var scratch = List[UInt8]()
        while i < n and bytes[i] != _SLASH:
            if bytes[i] == _TILDE:
                if i + 1 < n and bytes[i + 1] == _ZERO:
                    scratch.append(_TILDE)
                    i += 2
                    continue
                if i + 1 < n and bytes[i + 1] == _ONE:
                    scratch.append(_SLASH)
                    i += 2
                    continue
                raise Error(
                    "JSON Pointer escape must be '~0' or '~1': " + pointer
                )
            # Copied as a byte rather than through `chr`, which would
            # read a byte above 0x7F as a code point and re-encode it
            # as two, corrupting every non-ASCII key.
            scratch.append(bytes[i])
            i += 1
        tokens.append(String(unsafe_from_utf8=Span(scratch)))
        if i >= n:
            break
        i += 1

    return tokens^


def escape_token(token: String) -> String:
    """A reference token encoded for use inside a JSON Pointer.

    The inverse of what `parse_pointer` decodes: `~` becomes `~0` and
    `/` becomes `~1`, in that order, so the result decodes back to the
    token it came from.
    """
    var out = List[UInt8]()
    var bytes = token.as_bytes()
    for i in range(len(bytes)):
        var byte = bytes[i]
        if byte == _TILDE:
            out.append(_TILDE)
            out.append(_ZERO)
        elif byte == _SLASH:
            out.append(_TILDE)
            out.append(_ONE)
        else:
            out.append(byte)
    return String(unsafe_from_utf8=Span(out))


def build_pointer(tokens: List[String]) -> String:
    """A JSON Pointer naming the location reached by `tokens`."""
    var out = String()
    for i in range(len(tokens)):
        out += "/"
        out += escape_token(tokens[i])
    return out^


@always_inline
def is_dash(token: String) -> Bool:
    """Whether `token` is the past-the-end token `-`.

    RFC 6901 gives `-` a meaning only where a new element may be
    added, so the caller decides whether to honour it or reject it.
    """
    return token == "-"


def array_index(token: String, count: Int) raises -> Int:
    """A reference token read as an array index.

    RFC 6901 spells an index as `0` or a non-zero digit followed by
    digits, so `01`, `+1`, `-1`, ` 1` and the empty string are all
    invalid rather than merely out of range. `atol` accepts every one
    of them, which is why this does the digit check itself.

    Args:
        token: The reference token.
        count: The array's length, used for the range check.

    Returns:
        The index, which is in `[0, count)`.

    Raises:
        Error: If the token is not a valid index, or names an element
            the array does not have.
    """
    var index = index_value(token)
    if index >= count:
        raise Error(
            "JSON Pointer array index out of range: "
            + token
            + " (length "
            + String(count)
            + ")"
        )
    return index


def index_value(token: String) raises -> Int:
    """A reference token read as an index, without a range check.

    Used where the bound differs from the array's length, as it does
    for the `add` operation of RFC 6902, which may name the position
    one past the last element.
    """
    var bytes = token.as_bytes()
    var n = len(bytes)
    if n == 0:
        raise Error("JSON Pointer array index is empty")
    if token == "-":
        raise Error(
            "JSON Pointer '-' names a position past the end, which is only"
            " valid where an element may be added"
        )
    if bytes[0] == _MINUS:
        raise Error("JSON Pointer array index cannot be negative: " + token)
    if bytes[0] == _ZERO and n > 1:
        raise Error("JSON Pointer array index has a leading zero: " + token)

    var index = 0
    for i in range(n):
        var byte = bytes[i]
        if byte < _ZERO or byte > _NINE:
            raise Error("JSON Pointer array index is not a number: " + token)
        # Indices far beyond any real array still have to stop rather
        # than wrap, so the accumulator is capped well below overflow.
        if index > 0x7FFFFFFF:
            raise Error("JSON Pointer array index is too large: " + token)
        index = index * 10 + Int(byte - _ZERO)
    return index
