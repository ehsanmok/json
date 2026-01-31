# mojson CPU backend - Optimized JSON parser
# High-performance parsing with optimized algorithms
# Target: Match simdjson performance

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
# Fast Parser - Optimized for maximum throughput
# =============================================================================


struct FastParser:
    """High-performance JSON parser.

    Key optimizations:
    - memcpy for bulk copy operations
    - 8-digit unrolled integer parsing
    - Tight loops without bounds checks in hot paths
    """

    var data: List[UInt8]
    var length: Int
    var pos: Int
    var raw_json: String  # Keep reference for lazy values

    fn __init__(out self, s: String):
        """Initialize parser - single copy of input for safe access."""
        var bytes_span = s.as_bytes()
        var n = len(bytes_span)

        # Single copy using memcpy
        self.data = List[UInt8](capacity=n)
        self.data.resize(n, 0)
        memcpy(
            dest=self.data.unsafe_ptr(), src=bytes_span.unsafe_ptr(), count=n
        )

        self.length = n
        self.pos = 0
        self.raw_json = s

    @always_inline
    fn peek(self) -> UInt8:
        """Peek current byte."""
        return self.data[self.pos]

    @always_inline
    fn at_end(self) -> Bool:
        """Check if at end."""
        return self.pos >= self.length

    @always_inline
    fn advance(mut self):
        """Advance by 1."""
        self.pos += 1

    @always_inline
    fn advance_n(mut self, n: Int):
        """Advance by n."""
        self.pos += n

    @always_inline
    fn skip_whitespace(mut self):
        """Skip whitespace - tight loop with ref for speed."""
        ref data = self.data
        var length = self.length
        while self.pos < length:
            var c = data[self.pos]
            if c != 0x20 and c != 0x09 and c != 0x0A and c != 0x0D:
                return
            self.pos += 1

    fn parse(mut self) raises -> Value:
        """Parse JSON and return a Value."""
        self.skip_whitespace()

        if self.at_end():
            from ..errors import json_parse_error

            raise Error(json_parse_error("Empty input", self.raw_json, 0))

        var result = self.parse_value()

        self.skip_whitespace()
        if not self.at_end():
            from ..errors import json_parse_error

            raise Error(
                json_parse_error(
                    "Unexpected content after JSON value",
                    self.raw_json,
                    self.pos,
                )
            )

        return result^

    fn parse_value(mut self) raises -> Value:
        """Parse any JSON value."""
        self.skip_whitespace()

        if self.at_end():
            from ..errors import json_parse_error

            raise Error(
                json_parse_error(
                    "Unexpected end of input", self.raw_json, self.pos
                )
            )

        var c = self.peek()

        if c == ord("n"):
            return self.parse_null()
        elif c == ord("t"):
            return self.parse_true()
        elif c == ord("f"):
            return self.parse_false()
        elif c == ord('"'):
            return self.parse_string()
        elif c == ord("-") or (c >= ord("0") and c <= ord("9")):
            return self.parse_number()
        elif c == ord("["):
            return self.parse_array()
        elif c == ord("{"):
            return self.parse_object()
        else:
            from ..errors import json_parse_error

            raise Error(
                json_parse_error(
                    "Unexpected character: " + chr(Int(c)),
                    self.raw_json,
                    self.pos,
                )
            )

    @always_inline
    fn parse_null(mut self) raises -> Value:
        """Parse 'null' literal."""
        if self.pos + 4 <= self.length:
            if (
                self.data[self.pos] == ord("n")
                and self.data[self.pos + 1] == ord("u")
                and self.data[self.pos + 2] == ord("l")
                and self.data[self.pos + 3] == ord("l")
            ):
                self.pos += 4
                return Value(Null())
        from ..errors import json_parse_error

        raise Error(json_parse_error("Invalid 'null'", self.raw_json, self.pos))

    @always_inline
    fn parse_true(mut self) raises -> Value:
        """Parse 'true' literal."""
        if self.pos + 4 <= self.length:
            if (
                self.data[self.pos] == ord("t")
                and self.data[self.pos + 1] == ord("r")
                and self.data[self.pos + 2] == ord("u")
                and self.data[self.pos + 3] == ord("e")
            ):
                self.pos += 4
                return Value(True)
        from ..errors import json_parse_error

        raise Error(json_parse_error("Invalid 'true'", self.raw_json, self.pos))

    @always_inline
    fn parse_false(mut self) raises -> Value:
        """Parse 'false' literal."""
        if self.pos + 5 <= self.length:
            if (
                self.data[self.pos] == ord("f")
                and self.data[self.pos + 1] == ord("a")
                and self.data[self.pos + 2] == ord("l")
                and self.data[self.pos + 3] == ord("s")
                and self.data[self.pos + 4] == ord("e")
            ):
                self.pos += 5
                return Value(False)
        from ..errors import json_parse_error

        raise Error(
            json_parse_error("Invalid 'false'", self.raw_json, self.pos)
        )

    fn parse_string(mut self) raises -> Value:
        """Parse JSON string."""
        self.advance()  # Skip opening quote
        var start = self.pos
        var has_escapes = False
        ref data = self.data
        var length = self.length

        # Scan for string end
        while self.pos < length:
            var c = data[self.pos]
            if c == 0x22:  # Quote - end of string
                var end = self.pos
                self.pos += 1
                return self.build_string(start, end, has_escapes)
            elif c == 0x5C:  # Backslash
                has_escapes = True
                self.pos += 1
                if self.pos < length:
                    var esc = data[self.pos]
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
                                "Invalid escape sequence",
                                self.raw_json,
                                self.pos - 1,
                            )
                        )
                    self.pos += 1
            elif c < 0x20:  # Control character
                from ..errors import json_parse_error

                raise Error(
                    json_parse_error(
                        "Invalid control character in string",
                        self.raw_json,
                        self.pos,
                    )
                )
            else:
                self.pos += 1

        from ..errors import json_parse_error

        raise Error(
            json_parse_error("Unterminated string", self.raw_json, start - 1)
        )

    fn build_string(
        self, start: Int, end: Int, has_escapes: Bool
    ) raises -> Value:
        """Build string value."""
        var str_len = end - start

        if not has_escapes:
            # Fast path: direct memcpy
            var str_bytes = List[UInt8](capacity=str_len)
            str_bytes.resize(str_len, 0)
            memcpy(
                dest=str_bytes.unsafe_ptr(),
                src=self.data.unsafe_ptr() + start,
                count=str_len,
            )
            return Value(String(unsafe_from_utf8=str_bytes^))
        else:
            # Slow path: handle escapes
            var unescaped = unescape_json_string(self.data, start, end)
            return Value(String(unsafe_from_utf8=unescaped^))

    fn parse_number(mut self) raises -> Value:
        """Parse JSON number."""
        ref data = self.data
        var length = self.length
        var start = self.pos
        var is_float = False
        var negative = False

        # Optional minus
        if data[self.pos] == ord("-"):
            negative = True
            self.pos += 1

        if self.pos >= length:
            from ..errors import json_parse_error

            raise Error(
                json_parse_error("Invalid number", self.raw_json, start)
            )

        var c = data[self.pos]

        # Integer part
        if c == ord("0"):
            self.pos += 1
            if (
                self.pos < length
                and data[self.pos] >= ord("0")
                and data[self.pos] <= ord("9")
            ):
                from ..errors import json_parse_error

                raise Error(
                    json_parse_error(
                        "Leading zeros not allowed", self.raw_json, start
                    )
                )
        elif c >= ord("0") and c <= ord("9"):
            while self.pos < length:
                c = data[self.pos]
                if c < ord("0") or c > ord("9"):
                    break
                self.pos += 1
        else:
            from ..errors import json_parse_error

            raise Error(
                json_parse_error("Invalid number", self.raw_json, start)
            )

        # Fractional part
        if self.pos < length and data[self.pos] == ord("."):
            is_float = True
            self.pos += 1
            if (
                self.pos >= length
                or data[self.pos] < ord("0")
                or data[self.pos] > ord("9")
            ):
                from ..errors import json_parse_error

                raise Error(
                    json_parse_error(
                        "Expected digit after decimal point",
                        self.raw_json,
                        self.pos,
                    )
                )
            while self.pos < length:
                c = data[self.pos]
                if c < ord("0") or c > ord("9"):
                    break
                self.pos += 1

        # Exponent part
        if self.pos < length and (
            data[self.pos] == ord("e") or data[self.pos] == ord("E")
        ):
            is_float = True
            self.pos += 1
            if self.pos < length and (
                data[self.pos] == ord("+") or data[self.pos] == ord("-")
            ):
                self.pos += 1
            if (
                self.pos >= length
                or data[self.pos] < ord("0")
                or data[self.pos] > ord("9")
            ):
                from ..errors import json_parse_error

                raise Error(
                    json_parse_error(
                        "Expected digit in exponent", self.raw_json, self.pos
                    )
                )
            while self.pos < length:
                c = data[self.pos]
                if c < ord("0") or c > ord("9"):
                    break
                self.pos += 1

        if is_float:
            # Float: use string conversion
            var num_len = self.pos - start
            var num_bytes = List[UInt8](capacity=num_len)
            num_bytes.resize(num_len, 0)
            memcpy(
                dest=num_bytes.unsafe_ptr(),
                src=self.data.unsafe_ptr() + start,
                count=num_len,
            )
            var num_str = String(unsafe_from_utf8=num_bytes^)
            return Value(atof(num_str))
        else:
            # Integer: fast direct parsing with 8-digit unrolling
            var digit_start = start + 1 if negative else start
            return Value(self.parse_int_fast(digit_start, self.pos, negative))

    @always_inline
    fn parse_int_fast(self, start: Int, end: Int, negative: Bool) -> Int64:
        """Parse integer using 8-digit unrolling."""
        var result: Int64 = 0
        var pos = start

        # Unrolled 8-digit processing
        while pos + 8 <= end:
            var d0 = Int64(self.data[pos] - ord("0"))
            var d1 = Int64(self.data[pos + 1] - ord("0"))
            var d2 = Int64(self.data[pos + 2] - ord("0"))
            var d3 = Int64(self.data[pos + 3] - ord("0"))
            var d4 = Int64(self.data[pos + 4] - ord("0"))
            var d5 = Int64(self.data[pos + 5] - ord("0"))
            var d6 = Int64(self.data[pos + 6] - ord("0"))
            var d7 = Int64(self.data[pos + 7] - ord("0"))
            result = (
                result * 100000000
                + d0 * 10000000
                + d1 * 1000000
                + d2 * 100000
                + d3 * 10000
                + d4 * 1000
                + d5 * 100
                + d6 * 10
                + d7
            )
            pos += 8

        # Handle remaining 4 digits
        if pos + 4 <= end:
            var d0 = Int64(self.data[pos] - ord("0"))
            var d1 = Int64(self.data[pos + 1] - ord("0"))
            var d2 = Int64(self.data[pos + 2] - ord("0"))
            var d3 = Int64(self.data[pos + 3] - ord("0"))
            result = result * 10000 + d0 * 1000 + d1 * 100 + d2 * 10 + d3
            pos += 4

        # Handle remaining digits
        while pos < end:
            result = result * 10 + Int64(self.data[pos] - ord("0"))
            pos += 1

        return -result if negative else result

    fn parse_array(mut self) raises -> Value:
        """Parse JSON array."""
        ref data = self.data
        var length = self.length
        var array_start = self.pos
        self.advance()  # Skip '['
        self.skip_whitespace()

        # Empty array
        if self.pos < length and data[self.pos] == ord("]"):
            self.advance()
            return make_array_value("[]", 0)

        # Count elements by scanning
        var count = 0
        var depth = 1
        var scan_pos = self.pos
        var last_was_comma = False

        while scan_pos < length and depth > 0:
            var c = data[scan_pos]

            # Skip whitespace
            if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
                scan_pos += 1
                continue

            if c == 0x22:  # Quote - skip string
                last_was_comma = False
                scan_pos += 1
                while scan_pos < length:
                    var sc = data[scan_pos]
                    if sc == 0x5C:  # Backslash
                        scan_pos += 2
                        continue
                    if sc == 0x22:  # Quote
                        scan_pos += 1
                        break
                    scan_pos += 1
                continue
            elif c == 0x5B or c == 0x7B:  # [ or {
                last_was_comma = False
                depth += 1
            elif c == 0x5D or c == 0x7D:  # ] or }
                if depth == 1 and c == 0x5D and last_was_comma:
                    from ..errors import json_parse_error

                    raise Error(
                        json_parse_error(
                            "Trailing comma in array",
                            self.raw_json,
                            scan_pos - 1,
                        )
                    )
                depth -= 1
            elif c == 0x2C and depth == 1:  # Comma at depth 1
                if last_was_comma:
                    from ..errors import json_parse_error

                    raise Error(
                        json_parse_error(
                            "Double comma in array", self.raw_json, scan_pos
                        )
                    )
                last_was_comma = True
                count += 1
                scan_pos += 1
                continue
            else:
                last_was_comma = False

            scan_pos += 1

        if depth > 0:
            from ..errors import json_parse_error

            raise Error(
                json_parse_error(
                    "Unterminated array", self.raw_json, array_start
                )
            )

        count += 1  # Last element

        # Extract raw JSON using memcpy
        var array_end = scan_pos
        var raw_len = array_end - array_start
        var raw_bytes = List[UInt8](capacity=raw_len)
        raw_bytes.resize(raw_len, 0)
        memcpy(
            dest=raw_bytes.unsafe_ptr(),
            src=self.data.unsafe_ptr() + array_start,
            count=raw_len,
        )
        var raw = String(unsafe_from_utf8=raw_bytes^)

        self.pos = array_end
        return make_array_value(raw, count)

    fn parse_object(mut self) raises -> Value:
        """Parse JSON object."""
        var object_start = self.pos
        self.advance()  # Skip '{'
        self.skip_whitespace()

        # Empty object
        if self.pos < self.length and self.data[self.pos] == ord("}"):
            self.advance()
            var empty_keys = List[String]()
            return make_object_value("{}", empty_keys^)

        # Extract keys and count
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
            if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
                scan_pos += 1
                continue

            # Check for unquoted key
            if depth == 1 and expect_key and c != 0x22 and c != 0x7D:
                from ..errors import json_parse_error

                raise Error(
                    json_parse_error(
                        "Object key must be a string", self.raw_json, scan_pos
                    )
                )

            if c == 0x22:  # Quote
                var str_start = scan_pos + 1
                scan_pos += 1

                # Find end of string
                while scan_pos < self.length:
                    var sc = self.data[scan_pos]
                    if sc == 0x5C:  # Backslash
                        scan_pos += 2
                        continue
                    if sc == 0x22:  # Quote
                        break
                    scan_pos += 1

                # Extract key if at depth 1 and expecting key
                if depth == 1 and expect_key:
                    var key_len = scan_pos - str_start
                    var key_bytes = List[UInt8](capacity=key_len)
                    key_bytes.resize(key_len, 0)
                    memcpy(
                        dest=key_bytes.unsafe_ptr(),
                        src=self.data.unsafe_ptr() + str_start,
                        count=key_len,
                    )
                    keys.append(String(unsafe_from_utf8=key_bytes^))
                    expect_key = False

                    # Check for colon
                    var check_pos = scan_pos + 1
                    while check_pos < self.length and (
                        self.data[check_pos] == 0x20
                        or self.data[check_pos] == 0x09
                        or self.data[check_pos] == 0x0A
                        or self.data[check_pos] == 0x0D
                    ):
                        check_pos += 1
                    if (
                        check_pos >= self.length or self.data[check_pos] != 0x3A
                    ):  # :
                        from ..errors import json_parse_error

                        raise Error(
                            json_parse_error(
                                "Expected ':' after object key",
                                self.raw_json,
                                check_pos,
                            )
                        )
                elif depth == 1 and expect_value:
                    has_value_after_colon = True

                scan_pos += 1
                last_was_comma = False
                continue

            if c == 0x3A and depth == 1:  # :
                expect_key = False
                expect_value = True
                has_value_after_colon = False
                scan_pos += 1
                last_was_comma = False
                continue

            if c == 0x2C and depth == 1:  # ,
                if expect_value and not has_value_after_colon:
                    from ..errors import json_parse_error

                    raise Error(
                        json_parse_error(
                            "Expected value after ':'", self.raw_json, scan_pos
                        )
                    )
                expect_key = True
                expect_value = False
                last_was_comma = True
                scan_pos += 1
                continue

            if c == 0x7B or c == 0x5B:  # { or [
                if depth == 1 and expect_value:
                    has_value_after_colon = True
                depth += 1
                last_was_comma = False
            elif c == 0x7D or c == 0x5D:  # } or ]
                if depth == 1 and c == 0x7D:
                    if last_was_comma:
                        from ..errors import json_parse_error

                        raise Error(
                            json_parse_error(
                                "Trailing comma in object",
                                self.raw_json,
                                scan_pos - 1,
                            )
                        )
                    if expect_value and not has_value_after_colon:
                        from ..errors import json_parse_error

                        raise Error(
                            json_parse_error(
                                "Expected value after ':'",
                                self.raw_json,
                                scan_pos,
                            )
                        )
                depth -= 1
                last_was_comma = False
            else:
                if depth == 1 and expect_value:
                    has_value_after_colon = True
                last_was_comma = False

            scan_pos += 1

        if depth > 0:
            from ..errors import json_parse_error

            raise Error(
                json_parse_error(
                    "Unterminated object", self.raw_json, object_start
                )
            )

        # Extract raw JSON using memcpy
        var object_end = scan_pos
        var raw_len = object_end - object_start
        var raw_bytes = List[UInt8](capacity=raw_len)
        raw_bytes.resize(raw_len, 0)
        memcpy(
            dest=raw_bytes.unsafe_ptr(),
            src=self.data.unsafe_ptr() + object_start,
            count=raw_len,
        )
        var raw = String(unsafe_from_utf8=raw_bytes^)

        self.pos = object_end
        return make_object_value(raw, keys^)


# =============================================================================
# Public API
# =============================================================================


fn parse_simd(s: String) raises -> Value:
    """Parse JSON using optimized backend.

    Args:
        s: JSON string to parse.

    Returns:
        Parsed Value.

    Raises:
        Error on invalid JSON.
    """
    var parser = FastParser(s)
    return parser.parse()
