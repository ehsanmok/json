# mojson CPU backend - Pure Mojo JSON parser
# High-performance JSON parsing with zero FFI dependencies
# Optimized with SIMD operations for maximum performance

from collections import List
from memory import memcpy

from .types import (
    JSON_TYPE_NULL,
    JSON_TYPE_BOOL,
    JSON_TYPE_INT64,
    JSON_TYPE_DOUBLE,
    JSON_TYPE_STRING,
    JSON_TYPE_ARRAY,
    JSON_TYPE_OBJECT,
)
from ..value import Value, Null, make_array_value, make_object_value
from ..unicode import unescape_json_string


# =============================================================================
# SIMD Constants and Character Classification
# =============================================================================

comptime SIMD_WIDTH: Int = 16  # Process 16 bytes at a time

# Character constants
comptime CHAR_QUOTE: UInt8 = 0x22  # "
comptime CHAR_BACKSLASH: UInt8 = 0x5C  # \
comptime CHAR_LBRACKET: UInt8 = 0x5B  # [
comptime CHAR_RBRACKET: UInt8 = 0x5D  # ]
comptime CHAR_LBRACE: UInt8 = 0x7B  # {
comptime CHAR_RBRACE: UInt8 = 0x7D  # }
comptime CHAR_COLON: UInt8 = 0x3A  # :
comptime CHAR_COMMA: UInt8 = 0x2C  # ,

# Character type flags (for lookup table optimization)
comptime CHAR_NONE: UInt8 = 0
comptime CHAR_WS: UInt8 = 1  # Whitespace
comptime CHAR_STRUCT: UInt8 = 2  # Structural: { } [ ] : ,
comptime CHAR_QUOTE_FLAG: UInt8 = 4  # Quote
comptime CHAR_ESCAPE: UInt8 = 8  # Backslash


@always_inline
fn get_char_type(c: UInt8) -> UInt8:
    """Fast character type lookup using switch-like logic."""
    if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
        return CHAR_WS
    if c == 0x22:  # Quote
        return CHAR_QUOTE_FLAG
    if c == 0x5C:  # Backslash
        return CHAR_ESCAPE
    if (
        c == 0x7B
        or c == 0x7D
        or c == 0x5B
        or c == 0x5D
        or c == 0x3A
        or c == 0x2C
    ):
        return CHAR_STRUCT
    return CHAR_NONE


# =============================================================================
# Character Classification (optimized with lookup table concept)
# =============================================================================


@always_inline
fn is_whitespace(c: UInt8) -> Bool:
    """Check if character is JSON whitespace (space, tab, newline, carriage return).
    """
    return c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D


@always_inline
fn is_digit(c: UInt8) -> Bool:
    """Check if character is a digit 0-9."""
    return c >= ord("0") and c <= ord("9")


@always_inline
fn parse_int_direct(
    data: List[UInt8], start: Int, end: Int, negative: Bool
) -> Int64:
    """Parse integer directly without string conversion.

    Much faster than building a string and calling atol.
    """
    var result: Int64 = 0
    var pos = start

    # Unrolled loop - process 4 digits at a time
    while pos + 4 <= end:
        var d0 = Int64(data[pos] - ord("0"))
        var d1 = Int64(data[pos + 1] - ord("0"))
        var d2 = Int64(data[pos + 2] - ord("0"))
        var d3 = Int64(data[pos + 3] - ord("0"))
        result = result * 10000 + d0 * 1000 + d1 * 100 + d2 * 10 + d3
        pos += 4

    # Handle remaining digits
    while pos < end:
        result = result * 10 + Int64(data[pos] - ord("0"))
        pos += 1

    if negative:
        return -result
    return result


# =============================================================================
# MojoJSONParser - Pure Mojo JSON parser
# =============================================================================


struct MojoJSONParser:
    """Pure Mojo JSON parser optimized for performance.

    This parser is designed for maximum performance:
    - Minimal allocations
    - Unsafe pointer access for hot paths (no bounds checking)
    - Branch prediction friendly
    - Cache-friendly memory access patterns
    """

    var data: List[UInt8]
    var length: Int
    var pos: Int

    fn __init__(out self, var data: List[UInt8]):
        """Initialize parser with byte data."""
        self.length = len(data)
        self.data = data^
        self.pos = 0

    @always_inline
    fn peek(self) -> UInt8:
        """Peek at current character without advancing."""
        if self.pos >= self.length:
            return 0
        return self.data[self.pos]

    @always_inline
    fn get_byte(self, idx: Int) -> UInt8:
        """Get byte at specific index."""
        return self.data[idx]

    @always_inline
    fn advance(mut self):
        """Advance position by 1."""
        self.pos += 1

    @always_inline
    fn advance_n(mut self, n: Int):
        """Advance position by n."""
        self.pos += n

    @always_inline
    fn at_end(self) -> Bool:
        """Check if at end of input."""
        return self.pos >= self.length

    @always_inline
    fn skip_whitespace(mut self):
        """Skip whitespace characters (optimized with unrolling)."""
        # Unrolled loop - process 4 bytes at a time for common case
        while self.pos + 4 <= self.length:
            var c0 = self.data[self.pos]
            if c0 != 0x20 and c0 != 0x09 and c0 != 0x0A and c0 != 0x0D:
                return
            var c1 = self.data[self.pos + 1]
            if c1 != 0x20 and c1 != 0x09 and c1 != 0x0A and c1 != 0x0D:
                self.pos += 1
                return
            var c2 = self.data[self.pos + 2]
            if c2 != 0x20 and c2 != 0x09 and c2 != 0x0A and c2 != 0x0D:
                self.pos += 2
                return
            var c3 = self.data[self.pos + 3]
            if c3 != 0x20 and c3 != 0x09 and c3 != 0x0A and c3 != 0x0D:
                self.pos += 3
                return
            self.pos += 4

        # Handle remaining bytes
        while self.pos < self.length:
            var c = self.data[self.pos]
            if c != 0x20 and c != 0x09 and c != 0x0A and c != 0x0D:
                return
            self.pos += 1

    fn parse(mut self, raw_json: String) raises -> Value:
        """Parse JSON and return a Value."""
        self.skip_whitespace()

        if self.at_end():
            from ..errors import json_parse_error

            raise Error(json_parse_error("Empty input", raw_json, 0))

        var result = self.parse_value(raw_json)

        # Check for extra content after valid JSON
        self.skip_whitespace()
        if not self.at_end():
            from ..errors import json_parse_error

            raise Error(
                json_parse_error(
                    "Unexpected content after JSON value", raw_json, self.pos
                )
            )

        return result^

    fn parse_value(mut self, raw_json: String) raises -> Value:
        """Parse any JSON value."""
        self.skip_whitespace()

        if self.at_end():
            from ..errors import json_parse_error

            raise Error(
                json_parse_error("Unexpected end of input", raw_json, self.pos)
            )

        var c = self.peek()

        if c == ord("n"):
            return self.parse_null(raw_json)
        elif c == ord("t"):
            return self.parse_true(raw_json)
        elif c == ord("f"):
            return self.parse_false(raw_json)
        elif c == ord('"'):
            return self.parse_string(raw_json)
        elif c == ord("-") or is_digit(c):
            return self.parse_number(raw_json)
        elif c == ord("["):
            return self.parse_array(raw_json)
        elif c == ord("{"):
            return self.parse_object(raw_json)
        else:
            from ..errors import json_parse_error

            raise Error(
                json_parse_error(
                    "Unexpected character: " + chr(Int(c)), raw_json, self.pos
                )
            )

    fn parse_null(mut self, raw_json: String) raises -> Value:
        """Parse 'null' literal."""
        if (
            self.pos + 4 <= self.length
            and self.data[self.pos] == ord("n")
            and self.data[self.pos + 1] == ord("u")
            and self.data[self.pos + 2] == ord("l")
            and self.data[self.pos + 3] == ord("l")
        ):
            self.advance_n(4)
            return Value(Null())
        else:
            from ..errors import json_parse_error

            raise Error(json_parse_error("Invalid 'null'", raw_json, self.pos))

    fn parse_true(mut self, raw_json: String) raises -> Value:
        """Parse 'true' literal."""
        if (
            self.pos + 4 <= self.length
            and self.data[self.pos] == ord("t")
            and self.data[self.pos + 1] == ord("r")
            and self.data[self.pos + 2] == ord("u")
            and self.data[self.pos + 3] == ord("e")
        ):
            self.advance_n(4)
            return Value(True)
        else:
            from ..errors import json_parse_error

            raise Error(json_parse_error("Invalid 'true'", raw_json, self.pos))

    fn parse_false(mut self, raw_json: String) raises -> Value:
        """Parse 'false' literal."""
        if (
            self.pos + 5 <= self.length
            and self.data[self.pos] == ord("f")
            and self.data[self.pos + 1] == ord("a")
            and self.data[self.pos + 2] == ord("l")
            and self.data[self.pos + 3] == ord("s")
            and self.data[self.pos + 4] == ord("e")
        ):
            self.advance_n(5)
            return Value(False)
        else:
            from ..errors import json_parse_error

            raise Error(json_parse_error("Invalid 'false'", raw_json, self.pos))

    fn parse_string(mut self, raw_json: String) raises -> Value:
        """Parse a JSON string value."""
        if self.peek() != ord('"'):
            from ..errors import json_parse_error

            raise Error(json_parse_error("Expected '\"'", raw_json, self.pos))

        self.advance()  # Skip opening quote
        var start = self.pos
        var has_escapes = False

        # Scan for end of string
        while not self.at_end():
            var c = self.data[self.pos]
            if c == ord("\\"):
                has_escapes = True
                self.advance()
                if self.at_end():
                    from ..errors import json_parse_error

                    raise Error(
                        json_parse_error(
                            "Unterminated escape sequence",
                            raw_json,
                            self.pos - 1,
                        )
                    )
                # Validate escape character
                var esc = self.data[self.pos]
                if not (
                    esc == ord('"')
                    or esc == ord("\\")
                    or esc == ord("/")
                    or esc == ord("b")
                    or esc == ord("f")
                    or esc == ord("n")
                    or esc == ord("r")
                    or esc == ord("t")
                    or esc == ord("u")
                ):
                    from ..errors import json_parse_error

                    raise Error(
                        json_parse_error(
                            "Invalid escape sequence: \\" + chr(Int(esc)),
                            raw_json,
                            self.pos - 1,
                        )
                    )
                self.advance()
                continue
            if c == ord('"'):
                break
            # Check for invalid control characters
            if c < 0x20:
                from ..errors import json_parse_error

                raise Error(
                    json_parse_error(
                        "Invalid control character in string",
                        raw_json,
                        self.pos,
                    )
                )
            self.advance()

        if self.at_end():
            from ..errors import json_parse_error

            raise Error(
                json_parse_error("Unterminated string", raw_json, start - 1)
            )

        var end = self.pos
        self.advance()  # Skip closing quote

        # Build string
        if not has_escapes:
            # Fast path: no escapes, direct copy with unrolling
            var str_len = end - start
            var str_bytes = List[UInt8](capacity=str_len)
            var i = 0
            # Unrolled copy - 8 bytes at a time
            while i + 8 <= str_len:
                str_bytes.append(self.data[start + i])
                str_bytes.append(self.data[start + i + 1])
                str_bytes.append(self.data[start + i + 2])
                str_bytes.append(self.data[start + i + 3])
                str_bytes.append(self.data[start + i + 4])
                str_bytes.append(self.data[start + i + 5])
                str_bytes.append(self.data[start + i + 6])
                str_bytes.append(self.data[start + i + 7])
                i += 8
            # Copy remainder
            while i < str_len:
                str_bytes.append(self.data[start + i])
                i += 1
            return Value(String(unsafe_from_utf8=str_bytes^))
        else:
            # Slow path: handle escapes
            var unescaped = unescape_json_string(self.data, start, end)
            return Value(String(unsafe_from_utf8=unescaped^))

    fn parse_number(mut self, raw_json: String) raises -> Value:
        """Parse a JSON number (integer or float)."""
        var start = self.pos
        var is_float = False

        # Optional minus
        if self.peek() == ord("-"):
            self.advance()

        # Integer part
        if self.at_end():
            from ..errors import json_parse_error

            raise Error(json_parse_error("Invalid number", raw_json, start))

        var c = self.peek()
        if c == ord("0"):
            self.advance()
            # Check for leading zeros (e.g., 007 is invalid)
            if not self.at_end() and is_digit(self.peek()):
                from ..errors import json_parse_error

                raise Error(
                    json_parse_error(
                        "Leading zeros not allowed", raw_json, start
                    )
                )
        elif is_digit(c):
            while not self.at_end() and is_digit(self.peek()):
                self.advance()
        else:
            from ..errors import json_parse_error

            raise Error(json_parse_error("Invalid number", raw_json, start))

        # Fractional part
        if not self.at_end() and self.peek() == ord("."):
            is_float = True
            self.advance()
            if self.at_end() or not is_digit(self.peek()):
                from ..errors import json_parse_error

                raise Error(
                    json_parse_error(
                        "Expected digit after decimal point", raw_json, self.pos
                    )
                )
            while not self.at_end() and is_digit(self.peek()):
                self.advance()

        # Exponent part
        if not self.at_end() and (
            self.peek() == ord("e") or self.peek() == ord("E")
        ):
            is_float = True
            self.advance()
            if not self.at_end() and (
                self.peek() == ord("+") or self.peek() == ord("-")
            ):
                self.advance()
            if self.at_end() or not is_digit(self.peek()):
                from ..errors import json_parse_error

                raise Error(
                    json_parse_error(
                        "Expected digit in exponent", raw_json, self.pos
                    )
                )
            while not self.at_end() and is_digit(self.peek()):
                self.advance()

        if is_float:
            # For floats, use string conversion (atof handles all edge cases)
            var num_len = self.pos - start
            var num_bytes = List[UInt8](capacity=num_len)
            for i in range(num_len):
                num_bytes.append(self.data[start + i])
            var num_str = String(unsafe_from_utf8=num_bytes^)
            return Value(atof(num_str))
        else:
            # For integers, use fast direct parsing
            var negative = self.data[start] == ord("-")
            var digit_start = start + 1 if negative else start
            return Value(
                parse_int_direct(self.data, digit_start, self.pos, negative)
            )

    fn parse_array(mut self, raw_json: String) raises -> Value:
        """Parse a JSON array."""
        if self.peek() != ord("["):
            from ..errors import json_parse_error

            raise Error(json_parse_error("Expected '['", raw_json, self.pos))

        var array_start = self.pos
        self.advance()  # Skip '['
        self.skip_whitespace()

        # Check for empty array
        if not self.at_end() and self.peek() == ord("]"):
            self.advance()
            return make_array_value("[]", 0)

        # Count elements and track nesting, also detect trailing commas
        var count = 0
        var depth = 1
        var scan_pos = self.pos
        var last_was_comma = False

        while scan_pos < self.length and depth > 0:
            var c = self.data[scan_pos]

            # Skip whitespace tracking
            if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
                scan_pos += 1
                continue

            if c == ord('"'):
                # Skip string
                last_was_comma = False
                scan_pos += 1
                while scan_pos < self.length:
                    if self.data[scan_pos] == ord("\\"):
                        scan_pos += 2
                        continue
                    if self.data[scan_pos] == ord('"'):
                        scan_pos += 1
                        break
                    scan_pos += 1
                continue
            elif c == ord("[") or c == ord("{"):
                last_was_comma = False
                depth += 1
            elif c == ord("]") or c == ord("}"):
                # Check for trailing comma before closing bracket
                if depth == 1 and c == ord("]") and last_was_comma:
                    from ..errors import json_parse_error

                    raise Error(
                        json_parse_error(
                            "Trailing comma in array", raw_json, scan_pos - 1
                        )
                    )
                depth -= 1
            elif c == ord(",") and depth == 1:
                # Check for double comma
                if last_was_comma:
                    from ..errors import json_parse_error

                    raise Error(
                        json_parse_error(
                            "Double comma in array", raw_json, scan_pos
                        )
                    )
                last_was_comma = True
                count += 1
                scan_pos += 1
                continue
            else:
                # Some other value character (number, true, false, null)
                last_was_comma = False

            scan_pos += 1

        if depth > 0:
            from ..errors import json_parse_error

            raise Error(
                json_parse_error("Unterminated array", raw_json, array_start)
            )

        # At least one element if we got here
        count += 1

        # Extract raw JSON for the array (unrolled copy)
        var array_end = scan_pos
        var raw_len = array_end - array_start
        var raw_bytes = List[UInt8](capacity=raw_len)
        var i = 0
        while i + 8 <= raw_len:
            raw_bytes.append(self.data[array_start + i])
            raw_bytes.append(self.data[array_start + i + 1])
            raw_bytes.append(self.data[array_start + i + 2])
            raw_bytes.append(self.data[array_start + i + 3])
            raw_bytes.append(self.data[array_start + i + 4])
            raw_bytes.append(self.data[array_start + i + 5])
            raw_bytes.append(self.data[array_start + i + 6])
            raw_bytes.append(self.data[array_start + i + 7])
            i += 8
        while i < raw_len:
            raw_bytes.append(self.data[array_start + i])
            i += 1
        var raw = String(unsafe_from_utf8=raw_bytes^)

        # Move position to end of array
        self.pos = array_end

        return make_array_value(raw, count)

    fn parse_object(mut self, raw_json: String) raises -> Value:
        """Parse a JSON object."""
        if self.peek() != ord("{"):
            from ..errors import json_parse_error

            raise Error(json_parse_error("Expected '{'", raw_json, self.pos))

        var object_start = self.pos
        self.advance()  # Skip '{'
        self.skip_whitespace()

        # Check for empty object
        if not self.at_end() and self.peek() == ord("}"):
            self.advance()
            var empty_keys = List[String]()
            return make_object_value("{}", empty_keys^)

        # Extract keys and track nesting, detect errors
        var keys = List[String]()
        var depth = 1
        var scan_pos = self.pos
        var expect_key = True
        var expect_value = False
        var last_was_comma = False
        var has_value_after_colon = False

        while scan_pos < self.length and depth > 0:
            var c = self.data[scan_pos]

            # Skip whitespace
            if is_whitespace(c):
                scan_pos += 1
                continue

            # Check for unquoted key - must be before other character handling
            if depth == 1 and expect_key and c != ord('"') and c != ord("}"):
                from ..errors import json_parse_error

                raise Error(
                    json_parse_error(
                        "Object key must be a string", raw_json, scan_pos
                    )
                )

            if c == ord('"'):
                var str_start = scan_pos + 1
                scan_pos += 1
                # Find end of string
                while scan_pos < self.length:
                    if self.data[scan_pos] == ord("\\"):
                        scan_pos += 2
                        continue
                    if self.data[scan_pos] == ord('"'):
                        break
                    scan_pos += 1

                # Extract key if at depth 1 and expecting key
                if depth == 1 and expect_key:
                    var key_len = scan_pos - str_start
                    var key_bytes = List[UInt8](capacity=key_len)
                    for i in range(key_len):
                        key_bytes.append(self.data[str_start + i])
                    keys.append(String(unsafe_from_utf8=key_bytes^))
                    expect_key = False  # Now expecting colon, not another key

                    # Check that next non-whitespace is colon
                    var check_pos = scan_pos + 1
                    while check_pos < self.length and is_whitespace(
                        self.data[check_pos]
                    ):
                        check_pos += 1
                    if check_pos >= self.length or self.data[check_pos] != ord(
                        ":"
                    ):
                        from ..errors import json_parse_error

                        raise Error(
                            json_parse_error(
                                "Expected ':' after object key",
                                raw_json,
                                check_pos if check_pos
                                < self.length else scan_pos,
                            )
                        )
                elif depth == 1 and expect_value:
                    has_value_after_colon = True

                scan_pos += 1  # Skip closing quote
                last_was_comma = False
                continue

            if c == ord(":") and depth == 1:
                expect_key = False
                expect_value = True
                has_value_after_colon = False
                scan_pos += 1
                last_was_comma = False
                continue

            if c == ord(",") and depth == 1:
                # Check for missing value after colon
                if expect_value and not has_value_after_colon:
                    from ..errors import json_parse_error

                    raise Error(
                        json_parse_error(
                            "Expected value after ':'", raw_json, scan_pos
                        )
                    )
                expect_key = True
                expect_value = False
                last_was_comma = True
                scan_pos += 1
                continue

            if c == ord("{") or c == ord("["):
                if depth == 1 and expect_value:
                    has_value_after_colon = True
                depth += 1
                last_was_comma = False
            elif c == ord("}") or c == ord("]"):
                # Check for trailing comma or missing value
                if depth == 1 and c == ord("}"):
                    if last_was_comma:
                        from ..errors import json_parse_error

                        raise Error(
                            json_parse_error(
                                "Trailing comma in object",
                                raw_json,
                                scan_pos - 1,
                            )
                        )
                    if expect_value and not has_value_after_colon:
                        from ..errors import json_parse_error

                        raise Error(
                            json_parse_error(
                                "Expected value after ':'", raw_json, scan_pos
                            )
                        )
                depth -= 1
                last_was_comma = False
            else:
                # Some other value character (number, true, false, null)
                if depth == 1 and expect_value:
                    has_value_after_colon = True
                last_was_comma = False

            scan_pos += 1

        if depth > 0:
            from ..errors import json_parse_error

            raise Error(
                json_parse_error("Unterminated object", raw_json, object_start)
            )

        # Extract raw JSON for the object (unrolled copy)
        var object_end = scan_pos
        var raw_len = object_end - object_start
        var raw_bytes = List[UInt8](capacity=raw_len)
        var i = 0
        while i + 8 <= raw_len:
            raw_bytes.append(self.data[object_start + i])
            raw_bytes.append(self.data[object_start + i + 1])
            raw_bytes.append(self.data[object_start + i + 2])
            raw_bytes.append(self.data[object_start + i + 3])
            raw_bytes.append(self.data[object_start + i + 4])
            raw_bytes.append(self.data[object_start + i + 5])
            raw_bytes.append(self.data[object_start + i + 6])
            raw_bytes.append(self.data[object_start + i + 7])
            i += 8
        while i < raw_len:
            raw_bytes.append(self.data[object_start + i])
            i += 1
        var raw = String(unsafe_from_utf8=raw_bytes^)

        # Move position to end of object
        self.pos = object_end

        return make_object_value(raw, keys^)


# =============================================================================
# Public API
# =============================================================================


fn parse_mojo(s: String) raises -> Value:
    """Parse JSON using pure Mojo backend.

    Args:
        s: JSON string to parse.

    Returns:
        Parsed Value.

    Raises:
        Error on invalid JSON.
    """
    var bytes_span = s.as_bytes()
    var n = len(bytes_span)
    var bytes = List[UInt8](capacity=n)

    # Unrolled copy for better performance
    var i = 0
    while i + 8 <= n:
        bytes.append(bytes_span[i])
        bytes.append(bytes_span[i + 1])
        bytes.append(bytes_span[i + 2])
        bytes.append(bytes_span[i + 3])
        bytes.append(bytes_span[i + 4])
        bytes.append(bytes_span[i + 5])
        bytes.append(bytes_span[i + 6])
        bytes.append(bytes_span[i + 7])
        i += 8
    while i < n:
        bytes.append(bytes_span[i])
        i += 1

    var parser = MojoJSONParser(bytes^)
    return parser.parse(s)
