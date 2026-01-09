# JSON Iterator - Navigate parsed JSON results
from collections import List
from memory import memcpy

from .types import (
    JSONResult,
    JSONInput,
    is_open_bracket,
    is_close_bracket,
    CHAR_QUOTE,
)


struct JSONIterator:
    """Iterator for traversing parsed JSON structure.

    Provides APIs to navigate the parsed JSON tree efficiently.
    The pre-calculated pair_pos enables O(1) skipping of nested structures.
    """

    var result: JSONResult
    var input_data: List[UInt8]
    var current_pos: Int

    fn __init__(out self, var result: JSONResult, var input_data: List[UInt8]):
        """Initialize iterator with parsed result.

        Args:
            result: The parsed JSON result
            input_data: Original JSON data
        """
        self.result = result^
        self.input_data = input_data^
        self.current_pos = 0

    fn reset(mut self):
        """Reset iterator to beginning."""
        self.current_pos = 0

    fn size(self) -> Int:
        """Get input data size."""
        return len(self.input_data)

    fn goto_key(mut self, key: String) raises -> Int:
        """Move to the value associated with a key in the current object.

        Args:
            key: The key to search for

        Returns:
            The structural index of the value, or -1 if not found
        """
        if self.current_pos >= self.result.total_result_size():
            return -1

        var start_byte = Int(self.result.structural[self.current_pos])
        if start_byte >= self.size():
            return -1

        var c = self.input_data[start_byte]

        # Must be at an object
        if c != 0x7B:  # '{'
            return -1

        var end_idx = Int(self.result.pair_pos[self.current_pos])
        if end_idx < 0:
            end_idx = self.result.total_result_size()

        # Search for key by looking at bytes between structural positions
        var i = self.current_pos + 1  # Start after '{'

        while i < end_idx:
            var curr_byte = Int(self.result.structural[i])
            if curr_byte >= self.size():
                i += 1
                continue

            var curr_char = self.input_data[curr_byte]

            # If we hit a colon, the key is between previous structural and this colon
            if curr_char == 0x3A:  # ':'
                # Key is between prev structural position and colon
                var prev_byte: Int
                if i == self.current_pos + 1:
                    prev_byte = start_byte + 1  # Right after '{'
                else:
                    prev_byte = Int(self.result.structural[i - 1]) + 1

                # Find and extract the key string
                var key_start = prev_byte
                # Skip whitespace
                while key_start < curr_byte and self._is_whitespace(
                    self.input_data[key_start]
                ):
                    key_start += 1

                if (
                    key_start < curr_byte
                    and self.input_data[key_start] == CHAR_QUOTE
                ):
                    var extracted_key = self._extract_string(key_start)
                    if extracted_key == key:
                        # Found the key! Value starts after colon
                        self.current_pos = i  # Move to colon position
                        return self.current_pos

            # Skip nested structures
            if is_open_bracket(curr_char):
                var pair = Int(self.result.pair_pos[i])
                if pair > 0:
                    i = pair + 1
                    continue

            i += 1

        return -1

    fn _is_whitespace(self, c: UInt8) -> Bool:
        """Check if character is whitespace."""
        return c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D

    fn goto_array_index(mut self, index: Int) raises -> Int:
        """Move to element at index in the current array.

        Args:
            index: Zero-based index in the array

        Returns:
            The structural index of the element, or -1 if not found
        """
        if self.current_pos >= self.result.total_result_size():
            return -1

        var start_byte = Int(self.result.structural[self.current_pos])
        if start_byte >= self.size():
            return -1

        var c = self.input_data[start_byte]

        # Must be at an array
        if c != 0x5B:  # '['
            return -1

        var end_idx = Int(self.result.pair_pos[self.current_pos])
        if end_idx < 0:
            end_idx = self.result.total_result_size()

        # For arrays like [1,2,3], structural positions are: [ , , ]
        # Element 0 is between '[' and first ',' (or ']')
        # Element 1 is after first ',', etc.

        var current_idx = 0

        # Index 0: stay at '[', value read will look after it
        if index == 0:
            # Don't move - current_pos points to '['
            # get_value will handle extracting value after '['
            return self.current_pos

        # For index > 0, count commas
        var i = self.current_pos + 1
        while i < end_idx:
            var curr_byte = Int(self.result.structural[i])
            if curr_byte >= self.size():
                i += 1
                continue

            var curr_char = self.input_data[curr_byte]

            # Skip nested structures
            if is_open_bracket(curr_char):
                var pair = Int(self.result.pair_pos[i])
                if pair > 0:
                    current_idx += 1
                    if current_idx == index:
                        self.current_pos = i
                        return i
                    i = pair + 1
                    continue

            # Comma separates elements
            if curr_char == 0x2C:  # ','
                current_idx += 1
                if current_idx == index:
                    self.current_pos = i  # Point to comma before this element
                    return self.current_pos
            elif curr_char == 0x5D:  # ']'
                break  # End of array

            i += 1

        return -1

    fn goto_next_sibling(mut self) raises -> Int:
        """Move to the next sibling element.

        Returns:
            The structural index of the next sibling, or -1 if not found
        """
        if self.current_pos >= self.result.total_result_size():
            return -1

        var pos = Int(self.result.structural[self.current_pos])
        if pos >= self.size():
            return -1

        var c = self.input_data[pos]

        # If at an open bracket, skip to after its pair
        if is_open_bracket(c):
            var pair = Int(self.result.pair_pos[self.current_pos])
            if pair > 0:
                self.current_pos = pair + 1
                if self.current_pos < self.result.total_result_size():
                    return self.current_pos
                return -1

        # Otherwise, just move to next structural
        self.current_pos += 1
        if self.current_pos < self.result.total_result_size():
            return self.current_pos
        return -1

    fn get_value(self) raises -> String:
        """Get the value at the current position.

        Returns:
            The value as a string
        """
        if self.current_pos >= self.result.total_result_size():
            return ""

        var curr_byte = Int(self.result.structural[self.current_pos])
        if curr_byte >= self.size():
            return ""

        var c = self.input_data[curr_byte]

        # If at colon, value is after it
        if c == 0x3A:  # ':'
            var value_start = curr_byte + 1
            while value_start < self.size() and self._is_whitespace(
                self.input_data[value_start]
            ):
                value_start += 1

            if value_start >= self.size():
                return ""

            c = self.input_data[value_start]

            # String value
            if c == CHAR_QUOTE:
                return self._extract_string(value_start)

            # Object or array
            if is_open_bracket(c):
                if self.current_pos + 1 < self.result.total_result_size():
                    var next_struct = self.current_pos + 1
                    var pair_idx = Int(self.result.pair_pos[next_struct])
                    if (
                        pair_idx > 0
                        and pair_idx < self.result.total_result_size()
                    ):
                        var end_pos = Int(self.result.structural[pair_idx])
                        return self._extract_range(value_start, end_pos + 1)
                return self._extract_range(value_start, value_start + 1)

            # Primitive
            return self._extract_primitive(value_start)

        # If at comma, value is after it
        if c == 0x2C:  # ','
            var value_start = curr_byte + 1
            while value_start < self.size() and self._is_whitespace(
                self.input_data[value_start]
            ):
                value_start += 1

            if value_start >= self.size():
                return ""

            c = self.input_data[value_start]
            if c == CHAR_QUOTE:
                return self._extract_string(value_start)

            return self._extract_primitive(value_start)

        # For string values directly
        if c == CHAR_QUOTE:
            return self._extract_string(curr_byte)

        # For arrays, if at '[', get first element
        if c == 0x5B:  # '['
            var value_start = curr_byte + 1
            while value_start < self.size() and self._is_whitespace(
                self.input_data[value_start]
            ):
                value_start += 1
            if value_start < self.size():
                var vc = self.input_data[value_start]
                if vc == CHAR_QUOTE:
                    return self._extract_string(value_start)
                if is_open_bracket(vc):
                    # Nested structure - find matching bracket
                    if self.current_pos + 1 < self.result.total_result_size():
                        var next_struct = self.current_pos + 1
                        var pair_idx = Int(self.result.pair_pos[next_struct])
                        if (
                            pair_idx > 0
                            and pair_idx < self.result.total_result_size()
                        ):
                            var end_pos = Int(self.result.structural[pair_idx])
                            return self._extract_range(value_start, end_pos + 1)
                return self._extract_primitive(value_start)
            return ""

        # For objects, return the whole structure
        if c == 0x7B:  # '{'
            var pair = Int(self.result.pair_pos[self.current_pos])
            if pair < 0 or pair >= self.result.total_result_size():
                return ""
            var end_pos = Int(self.result.structural[pair])
            return self._extract_range(curr_byte, end_pos + 1)

        return self._extract_primitive(curr_byte)

    fn _extract_primitive(self, start: Int) raises -> String:
        """Extract a primitive value (number, true, false, null)."""
        var end_pos = start
        while end_pos < self.size():
            var ec = self.input_data[end_pos]
            if (
                ec == 0x2C
                or ec == 0x7D
                or ec == 0x5D
                or self._is_whitespace(ec)
            ):
                break
            end_pos += 1
        return self._extract_range(start, end_pos)

    fn _extract_string(self, start: Int) raises -> String:
        """Extract a string value (without quotes)."""
        if start >= self.size() or self.input_data[start] != CHAR_QUOTE:
            return ""

        # First pass: find end and count result length
        var i = start + 1
        var end_pos = i
        var has_escapes = False

        while end_pos < self.size():
            var c = self.input_data[end_pos]
            if c == 0x5C:  # backslash
                has_escapes = True
                end_pos += 2  # Skip escape sequence
                continue
            if c == CHAR_QUOTE:
                break
            end_pos += 1

        # Fast path: no escapes - use memcpy
        if not has_escapes:
            var length = end_pos - i
            if length <= 0:
                return ""
            var bytes = List[UInt8](capacity=length)
            bytes.resize(length, 0)
            memcpy(
                dest=bytes.unsafe_ptr(),
                src=self.input_data.unsafe_ptr() + i,
                count=length,
            )
            return String(unsafe_from_utf8=bytes^)

        # Slow path: handle escapes
        var bytes = List[UInt8](capacity=end_pos - i)
        var escaped = False

        while i < end_pos:
            var c = self.input_data[i]

            if escaped:
                if c == ord("n"):
                    bytes.append(0x0A)
                elif c == ord("t"):
                    bytes.append(0x09)
                elif c == ord("r"):
                    bytes.append(0x0D)
                elif c == ord("\\"):
                    bytes.append(0x5C)
                elif c == ord('"'):
                    bytes.append(0x22)
                else:
                    bytes.append(c)
                escaped = False
            elif c == 0x5C:
                escaped = True
            else:
                bytes.append(c)
            i += 1

        return String(unsafe_from_utf8=bytes^)

    fn _extract_range(self, start: Int, end: Int) raises -> String:
        """Extract a range of characters as a string."""
        if start >= self.size() or end > self.size() or start >= end:
            return ""

        var length = end - start
        var bytes = List[UInt8](capacity=length)
        bytes.resize(length, 0)
        # Use memcpy instead of byte-by-byte loop
        memcpy(
            dest=bytes.unsafe_ptr(),
            src=self.input_data.unsafe_ptr() + start,
            count=length,
        )
        return String(unsafe_from_utf8=bytes^)

    fn get_current_char(self) -> UInt8:
        """Get the character at current structural position."""
        if self.current_pos >= self.result.total_result_size():
            return 0

        var pos = Int(self.result.structural[self.current_pos])
        if pos >= self.size():
            return 0

        return self.input_data[pos]
