# json - On-demand (lazy) JSON parsing
#
# LazyValue defers parsing until values are actually accessed.
# This is efficient when you only need a few fields from a large document.

from .value import Value, Null
from .pointer import index_value, parse_pointer
from .value.raw_ops import _extract_field_value, _extract_array_element
from .cpu import parse_cpu_native_tape


struct LazyValue:
    """A lazily-parsed JSON value.

    Unlike Value, LazyValue stores the raw JSON and only parses
    when you access specific fields or elements. This is efficient
    for large documents where you only need a few values.

    Example:
        var lazy = loads_lazy(huge_json_string)
        var name = lazy.get("/users/0/name")
        Only parses the path to "users/0/name", not the whole document.
    """

    var _raw: String
    var _type: Int  # -1=unknown, 0=null, 1=bool, 2=int, 3=float, 4=string, 5=array, 6=object

    def __init__(out self, raw: String):
        """Create a LazyValue from raw JSON string."""
        self._raw = raw
        self._type = -1  # Unknown until accessed

    def _detect_type(mut self):
        """Detect the JSON type without full parsing."""
        if self._type >= 0:
            return

        var raw_bytes = self._raw.as_bytes()
        var n = len(raw_bytes)
        var i = 0

        # Skip whitespace
        while i < n and (
            raw_bytes[i] == UInt8(ord(" "))
            or raw_bytes[i] == UInt8(ord("\t"))
            or raw_bytes[i] == UInt8(ord("\n"))
            or raw_bytes[i] == UInt8(ord("\r"))
        ):
            i += 1

        if i >= n:
            self._type = 0  # Treat empty as null
            return

        var c = raw_bytes[i]
        if c == UInt8(ord("n")):
            self._type = 0  # null
        elif c == UInt8(ord("t")) or c == UInt8(ord("f")):
            self._type = 1  # bool
        elif c == UInt8(ord('"')):
            self._type = 4  # string
        elif c == UInt8(ord("[")):
            self._type = 5  # array
        elif c == UInt8(ord("{")):
            self._type = 6  # object
        elif c == UInt8(ord("-")) or (
            c >= UInt8(ord("0")) and c <= UInt8(ord("9"))
        ):
            # Determine if int or float
            self._type = 2  # Assume int
            while i < n:
                c = raw_bytes[i]
                if (
                    c == UInt8(ord("."))
                    or c == UInt8(ord("e"))
                    or c == UInt8(ord("E"))
                ):
                    self._type = 3  # float
                    break
                if (
                    c == UInt8(ord(","))
                    or c == UInt8(ord("}"))
                    or c == UInt8(ord("]"))
                    or c == UInt8(ord(" "))
                    or c == UInt8(ord("\n"))
                ):
                    break
                i += 1
        else:
            self._type = 0  # Default to null

    def is_null(mut self) -> Bool:
        """Check if value is null."""
        self._detect_type()
        return self._type == 0

    def is_bool(mut self) -> Bool:
        """Check if value is a boolean."""
        self._detect_type()
        return self._type == 1

    def is_int(mut self) -> Bool:
        """Check if value is an integer."""
        self._detect_type()
        return self._type == 2

    def is_float(mut self) -> Bool:
        """Check if value is a float."""
        self._detect_type()
        return self._type == 3

    def is_string(mut self) -> Bool:
        """Check if value is a string."""
        self._detect_type()
        return self._type == 4

    def is_array(mut self) -> Bool:
        """Check if value is an array."""
        self._detect_type()
        return self._type == 5

    def is_object(mut self) -> Bool:
        """Check if value is an object."""
        self._detect_type()
        return self._type == 6

    def raw(self) -> String:
        """Get the raw JSON string."""
        return self._raw

    def get(self, pointer: String) raises -> Value:
        """Get a value at the given JSON Pointer path.

        Only parses the necessary parts of the JSON to reach the target.

        Args:
            pointer: JSON Pointer path (e.g., "/users/0/name").

        Returns:
            The Value at the specified path.

        Example:
            var name = lazy.get("/users/0/name").
        """
        if pointer == "":
            return self.parse()

        if not pointer.startswith("/"):
            raise Error("JSON Pointer must start with '/' or be empty")

        return _lazy_navigate(self._raw, pointer)

    def get_string(self, pointer: String) raises -> String:
        """Get a string value at the given path.

        Args:
            pointer: JSON Pointer path.

        Returns:
            The string value.
        """
        var v = self.get(pointer)
        if not v.is_string():
            raise Error("Value at " + pointer + " is not a string")
        return v.string_value()

    def get_int(self, pointer: String) raises -> Int64:
        """Get an integer value at the given path.

        Args:
            pointer: JSON Pointer path.

        Returns:
            The integer value.
        """
        var v = self.get(pointer)
        if not v.is_int():
            raise Error("Value at " + pointer + " is not an integer")
        return v.int_value()

    def get_bool(self, pointer: String) raises -> Bool:
        """Get a boolean value at the given path.

        Args:
            pointer: JSON Pointer path.

        Returns:
            The boolean value.
        """
        var v = self.get(pointer)
        if not v.is_bool():
            raise Error("Value at " + pointer + " is not a boolean")
        return v.bool_value()

    def parse(self) raises -> Value:
        """Fully parse the JSON into a Value.

        Use this when you need the complete parsed structure.
        For accessing specific fields, use get() instead.
        """
        return parse_cpu_native_tape(self._raw.copy())

    def __getitem__(self, key: String) raises -> LazyValue:
        """Get object field as a new LazyValue.

        Args:
            key: Object key.

        Returns:
            LazyValue for the field (still lazy).
        """
        var value_str = _extract_field_value(self._raw, key)
        return LazyValue(value_str)

    def __getitem__(self, index: Int) raises -> LazyValue:
        """Get array element as a new LazyValue.

        Args:
            index: Array index.

        Returns:
            LazyValue for the element (still lazy).
        """
        var value_str = _extract_array_element(self._raw, index)
        return LazyValue(value_str)


def loads_lazy(s: String) -> LazyValue:
    """Create a lazy JSON value without parsing.

    The JSON is only parsed when you access specific values.
    This is efficient for large documents where you only need
    a few fields.

    Args:
        s: JSON string.

    Returns:
        LazyValue that parses on demand.

    Example:
        var lazy = loads_lazy(huge_json)
        var name = lazy.get("/users/0/name")  # Only parses path to name.
    """
    return LazyValue(s)


def _lazy_navigate(raw: String, pointer: String) raises -> Value:
    """Navigate to a value using JSON Pointer without full parsing."""
    var current_raw = raw
    # Tokens come from the shared RFC 6901 parser rather than a local
    # copy of it, which is what used to make `lazy.get("/")` return the
    # whole document while `Value.at("/")` looked for the member whose
    # key is the empty string.
    var tokens = parse_pointer(pointer)

    for token_index in range(len(tokens)):
        var token = tokens[token_index]

        # Determine if current is array or object
        var current_bytes = current_raw.as_bytes()
        var j = 0
        while j < len(current_bytes) and (
            current_bytes[j] == UInt8(ord(" "))
            or current_bytes[j] == UInt8(ord("\t"))
            or current_bytes[j] == UInt8(ord("\n"))
        ):
            j += 1

        if j >= len(current_bytes):
            raise Error("Invalid JSON in pointer navigation")

        var first = current_bytes[j]

        if first == UInt8(ord("{")):
            # Object - extract field
            current_raw = _extract_field_value(current_raw, token)
        elif first == UInt8(ord("[")):
            # Array - extract element by index
            current_raw = _extract_array_element(
                current_raw, index_value(token)
            )
        else:
            raise Error("Cannot navigate into primitive with pointer")

    return parse_cpu_native_tape(current_raw^)
