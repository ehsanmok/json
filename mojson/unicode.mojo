# mojson - Unicode handling for JSON

from collections import List


fn hex_digit_value(c: UInt8) -> Int:
    """Convert a hex digit character to its value.

    Returns -1 if not a valid hex digit.
    """
    if c >= ord("0") and c <= ord("9"):
        return Int(c - ord("0"))
    if c >= ord("a") and c <= ord("f"):
        return Int(c - ord("a") + 10)
    if c >= ord("A") and c <= ord("F"):
        return Int(c - ord("A") + 10)
    return -1


fn parse_unicode_escape(data: List[UInt8], start: Int) -> Int:
    """Parse a 4-digit hex unicode escape sequence.

    Args:
        data: The byte span containing the escape.
        start: Index of first hex digit (after backslash-u).

    Returns:
        The code point value, or -1 if invalid.
    """
    if start + 4 > len(data):
        return -1

    var result = 0
    for i in range(4):
        var digit = hex_digit_value(data[start + i])
        if digit < 0:
            return -1
        result = result * 16 + digit

    return result


fn is_high_surrogate(code_point: Int) -> Bool:
    """Check if code point is a high surrogate (U+D800 - U+DBFF)."""
    return code_point >= 0xD800 and code_point <= 0xDBFF


fn is_low_surrogate(code_point: Int) -> Bool:
    """Check if code point is a low surrogate (U+DC00 - U+DFFF)."""
    return code_point >= 0xDC00 and code_point <= 0xDFFF


fn decode_surrogate_pair(high: Int, low: Int) -> Int:
    """Decode a surrogate pair to a full code point.

    Args:
        high: High surrogate (U+D800 - U+DBFF).
        low: Low surrogate (U+DC00 - U+DFFF).

    Returns:
        The full Unicode code point (U+10000 - U+10FFFF).
    """
    return 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)


fn encode_utf8(code_point: Int, mut bytes: List[UInt8]):
    """Encode a Unicode code point as UTF-8 bytes.

    Args:
        code_point: Unicode code point (0 - 0x10FFFF).
        bytes: List to append UTF-8 bytes to.
    """
    if code_point < 0x80:
        # 1-byte sequence (ASCII)
        bytes.append(UInt8(code_point))
    elif code_point < 0x800:
        # 2-byte sequence
        bytes.append(UInt8(0xC0 | (code_point >> 6)))
        bytes.append(UInt8(0x80 | (code_point & 0x3F)))
    elif code_point < 0x10000:
        # 3-byte sequence
        bytes.append(UInt8(0xE0 | (code_point >> 12)))
        bytes.append(UInt8(0x80 | ((code_point >> 6) & 0x3F)))
        bytes.append(UInt8(0x80 | (code_point & 0x3F)))
    else:
        # 4-byte sequence
        bytes.append(UInt8(0xF0 | (code_point >> 18)))
        bytes.append(UInt8(0x80 | ((code_point >> 12) & 0x3F)))
        bytes.append(UInt8(0x80 | ((code_point >> 6) & 0x3F)))
        bytes.append(UInt8(0x80 | (code_point & 0x3F)))


fn unescape_json_string(data: List[UInt8], start: Int, end: Int) -> List[UInt8]:
    """Unescape a JSON string, handling all escape sequences including unicode.

    Args:
        data: The byte span containing the string content.
        start: Start index (after opening quote).
        end: End index (before closing quote).

    Returns:
        Unescaped bytes.
    """
    var result = List[UInt8](capacity=end - start)
    var i = start

    while i < end:
        var c = data[i]

        if c == ord("\\") and i + 1 < end:
            var next = data[i + 1]

            if next == ord("n"):
                result.append(0x0A)
                i += 2
            elif next == ord("t"):
                result.append(0x09)
                i += 2
            elif next == ord("r"):
                result.append(0x0D)
                i += 2
            elif next == ord("\\"):
                result.append(0x5C)
                i += 2
            elif next == ord('"'):
                result.append(0x22)
                i += 2
            elif next == ord("/"):
                result.append(0x2F)
                i += 2
            elif next == ord("b"):
                result.append(0x08)
                i += 2
            elif next == ord("f"):
                result.append(0x0C)
                i += 2
            elif next == ord("u"):
                # Unicode escape \uXXXX
                var code_point = parse_unicode_escape(data, i + 2)
                if code_point < 0:
                    # Invalid escape, keep as-is
                    result.append(c)
                    i += 1
                    continue

                # Check for surrogate pair
                if is_high_surrogate(code_point):
                    # Look for low surrogate
                    if (
                        i + 10 < end
                        and data[i + 6] == ord("\\")
                        and data[i + 7] == ord("u")
                    ):
                        var low = parse_unicode_escape(data, i + 8)
                        if is_low_surrogate(low):
                            # Decode surrogate pair
                            code_point = decode_surrogate_pair(code_point, low)
                            i += 12  # \uXXXX\uXXXX
                            encode_utf8(code_point, result)
                            continue
                    # Invalid surrogate - emit replacement character
                    result.append(0xEF)
                    result.append(0xBF)
                    result.append(0xBD)
                    i += 6
                elif is_low_surrogate(code_point):
                    # Orphan low surrogate - emit replacement character
                    result.append(0xEF)
                    result.append(0xBF)
                    result.append(0xBD)
                    i += 6
                else:
                    encode_utf8(code_point, result)
                    i += 6  # \uXXXX
            else:
                # Unknown escape, keep the backslash and char
                result.append(c)
                i += 1
        else:
            result.append(c)
            i += 1

    return result^
