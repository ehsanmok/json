# simdjson FFI wrapper for Mojo
# Provides high-performance JSON parsing via simdjson C++ library
# Uses OwnedDLHandle for runtime library loading

from std.ffi import OwnedDLHandle, external_call
from std.os import getenv
from std.memory import UnsafePointer, Span
from std.collections import List
from ..errors import json_parse_error, find_error_position


def _find_simdjson_library() -> String:
    """Find the simdjson wrapper library in standard locations."""
    # Check CONDA_PREFIX first (installed via conda/pixi)
    var conda_prefix = getenv("CONDA_PREFIX", "")
    if conda_prefix:
        return conda_prefix + "/lib/libsimdjson_wrapper.so"
    # Fallback to local build directory (development)
    return "build/libsimdjson_wrapper.so"


# Result codes from simdjson_wrapper.h
comptime SIMDJSON_OK: Int = 0
comptime SIMDJSON_ERROR_INVALID_JSON: Int = 1
comptime SIMDJSON_ERROR_CAPACITY: Int = 2
comptime SIMDJSON_ERROR_UTF8: Int = 3
comptime SIMDJSON_ERROR_OTHER: Int = 99

# Type codes from simdjson_wrapper.h
comptime SIMDJSON_TYPE_NULL: Int = 0
comptime SIMDJSON_TYPE_BOOL: Int = 1
comptime SIMDJSON_TYPE_INT64: Int = 2
comptime SIMDJSON_TYPE_UINT64: Int = 3
comptime SIMDJSON_TYPE_DOUBLE: Int = 4
comptime SIMDJSON_TYPE_STRING: Int = 5
comptime SIMDJSON_TYPE_ARRAY: Int = 6
comptime SIMDJSON_TYPE_OBJECT: Int = 7


struct SimdjsonFFI:
    """Low-level simdjson FFI bindings. All pointer args are passed as Int."""

    var _lib: OwnedDLHandle
    var _parser: Int  # Opaque pointer as Int

    # Parser functions
    var _create_parser: def() abi("C") -> Int
    var _destroy_parser: def(Int) abi("C") -> None
    var _parse: def(Int, Int, Int) abi("C") -> Int
    var _get_root: def(Int) abi("C") -> Int

    # Value functions
    var _value_get_type: def(Int) abi("C") -> Int
    var _value_get_bool: def(Int, Int) abi("C") -> Int
    var _value_get_int64: def(Int, Int) abi("C") -> Int
    var _value_get_uint64: def(Int, Int) abi("C") -> Int
    var _value_get_double: def(Int, Int) abi("C") -> Int
    var _value_get_string: def(Int, Int, Int) abi("C") -> Int
    var _value_free: def(Int) abi("C") -> None

    # Array functions
    var _array_begin: def(Int) abi("C") -> Int
    var _array_iter_done: def(Int) abi("C") -> Int
    var _array_iter_get: def(Int) abi("C") -> Int
    var _array_iter_next: def(Int) abi("C") -> None
    var _array_iter_free: def(Int) abi("C") -> None
    var _array_count: def(Int) abi("C") -> Int

    # Object functions
    var _object_begin: def(Int) abi("C") -> Int
    var _object_iter_done: def(Int) abi("C") -> Int
    var _object_iter_get_key: def(Int, Int, Int) abi("C") -> None
    var _object_iter_get_value: def(Int) abi("C") -> Int
    var _object_iter_next: def(Int) abi("C") -> None
    var _object_iter_free: def(Int) abi("C") -> None
    var _object_count: def(Int) abi("C") -> Int

    # Memory helper: copies n bytes from src_addr (integer) to dst (pointer as Int).
    # Avoids int-to-UnsafePointer construction in Mojo, which varies across versions.
    var _memcpy_from_addr: def(Int, Int, Int) abi("C") -> None

    def __init__(out self, lib_path: String = "") raises:
        """Initialize by loading the simdjson wrapper library.

        Args:
            lib_path: Path to the library. If empty, searches standard locations:
                      1. $CONDA_PREFIX/lib/libsimdjson_wrapper.so (installed).
                      2. build/libsimdjson_wrapper.so (development).
        """
        var path = lib_path if lib_path else _find_simdjson_library()
        self._lib = OwnedDLHandle(path)

        # Parser functions
        self._create_parser = self._lib.get_function[def() abi("C") -> Int](
            "simdjson_create_parser"
        )
        self._destroy_parser = self._lib.get_function[def(Int) abi("C") -> None](
            "simdjson_destroy_parser"
        )
        self._parse = self._lib.get_function[def(Int, Int, Int) abi("C") -> Int](
            "simdjson_parse"
        )
        self._get_root = self._lib.get_function[def(Int) abi("C") -> Int](
            "simdjson_get_root"
        )

        # Value functions
        self._value_get_type = self._lib.get_function[def(Int) abi("C") -> Int](
            "simdjson_value_get_type"
        )
        self._value_get_bool = self._lib.get_function[def(Int, Int) abi("C") -> Int](
            "simdjson_value_get_bool"
        )
        self._value_get_int64 = self._lib.get_function[def(Int, Int) abi("C") -> Int](
            "simdjson_value_get_int64"
        )
        self._value_get_uint64 = self._lib.get_function[def(Int, Int) abi("C") -> Int](
            "simdjson_value_get_uint64"
        )
        self._value_get_double = self._lib.get_function[def(Int, Int) abi("C") -> Int](
            "simdjson_value_get_double"
        )
        self._value_get_string = self._lib.get_function[
            def(Int, Int, Int) abi("C") -> Int
        ]("simdjson_value_get_string")
        self._value_free = self._lib.get_function[def(Int) abi("C") -> None](
            "simdjson_value_free"
        )

        # Array functions
        self._array_begin = self._lib.get_function[def(Int) abi("C") -> Int](
            "simdjson_array_begin"
        )
        self._array_iter_done = self._lib.get_function[def(Int) abi("C") -> Int](
            "simdjson_array_iter_done"
        )
        self._array_iter_get = self._lib.get_function[def(Int) abi("C") -> Int](
            "simdjson_array_iter_get"
        )
        self._array_iter_next = self._lib.get_function[def(Int) abi("C") -> None](
            "simdjson_array_iter_next"
        )
        self._array_iter_free = self._lib.get_function[def(Int) abi("C") -> None](
            "simdjson_array_iter_free"
        )
        self._array_count = self._lib.get_function[def(Int) abi("C") -> Int](
            "simdjson_array_count"
        )

        # Object functions
        self._object_begin = self._lib.get_function[def(Int) abi("C") -> Int](
            "simdjson_object_begin"
        )
        self._object_iter_done = self._lib.get_function[def(Int) abi("C") -> Int](
            "simdjson_object_iter_done"
        )
        self._object_iter_get_key = self._lib.get_function[
            def(Int, Int, Int) abi("C") -> None
        ]("simdjson_object_iter_get_key")
        self._object_iter_get_value = self._lib.get_function[def(Int) abi("C") -> Int](
            "simdjson_object_iter_get_value"
        )
        self._object_iter_next = self._lib.get_function[def(Int) abi("C") -> None](
            "simdjson_object_iter_next"
        )
        self._object_iter_free = self._lib.get_function[def(Int) abi("C") -> None](
            "simdjson_object_iter_free"
        )
        self._object_count = self._lib.get_function[def(Int) abi("C") -> Int](
            "simdjson_object_count"
        )
        self._memcpy_from_addr = self._lib.get_function[
            def(Int, Int, Int) abi("C") -> None
        ]("simdjson_memcpy_from_addr")

        # Create the parser
        self._parser = self._create_parser()
        if self._parser == 0:
            raise Error("Failed to create simdjson parser")

    def destroy(mut self):
        """Clean up the parser. Call this explicitly when done."""
        if self._parser != 0:
            self._destroy_parser(self._parser)
            self._parser = 0

    def parse(mut self, json: String) raises -> Int:
        """Parse JSON and return root value handle."""
        var json_copy = json
        var c_str = json_copy.as_c_string_slice()
        var ptr = Int(c_str.unsafe_ptr())
        var length = len(json_copy)

        var err = self._parse(self._parser, ptr, length)

        if err != SIMDJSON_OK:
            var pos = find_error_position(json)
            if err == SIMDJSON_ERROR_INVALID_JSON:
                raise Error(json_parse_error("Invalid JSON syntax", json, pos))
            elif err == SIMDJSON_ERROR_UTF8:
                raise Error(
                    json_parse_error("Invalid UTF-8 encoding", json, pos)
                )
            elif err == SIMDJSON_ERROR_CAPACITY:
                raise Error("JSON document too large (exceeds parser capacity)")
            else:
                raise Error(json_parse_error("Unknown parse error", json, pos))

        return self._get_root(self._parser)

    def get_type(self, value: Int) -> Int:
        """Get the type of a value."""
        return self._value_get_type(value)

    def get_bool(self, value: Int) raises -> Bool:
        """Get value as boolean."""
        var result = List[Int32](capacity=1)
        result.append(0)
        var err = self._value_get_bool(value, Int(result.unsafe_ptr()))
        if err != SIMDJSON_OK:
            raise Error("Value is not a boolean")
        return result[0] != 0

    def get_int(self, value: Int) raises -> Int64:
        """Get value as int64."""
        var result = List[Int64](capacity=1)
        result.append(0)
        var err = self._value_get_int64(value, Int(result.unsafe_ptr()))
        if err != SIMDJSON_OK:
            raise Error("Value is not an integer")
        return result[0]

    def get_uint(self, value: Int) raises -> UInt64:
        """Get value as uint64."""
        var result = List[UInt64](capacity=1)
        result.append(0)
        var err = self._value_get_uint64(value, Int(result.unsafe_ptr()))
        if err != SIMDJSON_OK:
            raise Error("Value is not an unsigned integer")
        return result[0]

    def get_float(self, value: Int) raises -> Float64:
        """Get value as double."""
        var result = List[Float64](capacity=1)
        result.append(0.0)
        var err = self._value_get_double(value, Int(result.unsafe_ptr()))
        if err != SIMDJSON_OK:
            raise Error("Value is not a float")
        return result[0]

    def get_string(self, value: Int) raises -> String:
        """Get value as string - uses unsafe_from_utf8 for zero-copy."""
        var data_ptr = List[Int](capacity=1)
        data_ptr.append(0)
        var len_buf = List[Int](capacity=1)
        len_buf.append(0)

        var err = self._value_get_string(
            value, Int(data_ptr.unsafe_ptr()), Int(len_buf.unsafe_ptr())
        )

        if err != SIMDJSON_OK:
            raise Error("Value is not a string")

        var addr = data_ptr[0]
        var length = len_buf[0]

        if length == 0:
            return String("")

        # Copy via C shim: avoids UnsafePointer-from-Int construction in Mojo.
        # simdjson guarantees valid UTF-8; unsafe_from_utf8 takes raw bytes.
        var bytes = List[UInt8](capacity=length)
        bytes.resize(length, 0)
        self._memcpy_from_addr(Int(bytes.unsafe_ptr()), addr, length)
        return String(unsafe_from_utf8=bytes^)

    def free_value(self, value: Int):
        """Free a value handle."""
        self._value_free(value)

    def array_count(self, value: Int) -> Int:
        """Get array element count."""
        return self._array_count(value)

    def array_begin(self, value: Int) -> Int:
        """Start iterating over array."""
        return self._array_begin(value)

    def array_iter_done(self, iter: Int) -> Bool:
        """Check if array iteration is done."""
        return self._array_iter_done(iter) != 0

    def array_iter_get(self, iter: Int) -> Int:
        """Get current array element."""
        return self._array_iter_get(iter)

    def array_iter_next(self, iter: Int):
        """Move to next array element."""
        self._array_iter_next(iter)

    def array_iter_free(self, iter: Int):
        """Free array iterator."""
        self._array_iter_free(iter)

    def object_count(self, value: Int) -> Int:
        """Get object key count."""
        return self._object_count(value)

    def object_begin(self, value: Int) -> Int:
        """Start iterating over object."""
        return self._object_begin(value)

    def object_iter_done(self, iter: Int) -> Bool:
        """Check if object iteration is done."""
        return self._object_iter_done(iter) != 0

    def object_iter_get_key(self, iter: Int) raises -> String:
        """Get current object key - uses unsafe_from_utf8 for zero-copy."""
        var data_ptr = List[Int](capacity=1)
        data_ptr.append(0)
        var len_buf = List[Int](capacity=1)
        len_buf.append(0)

        self._object_iter_get_key(
            iter, Int(data_ptr.unsafe_ptr()), Int(len_buf.unsafe_ptr())
        )

        var addr = data_ptr[0]
        var length = len_buf[0]

        if addr == 0:
            raise Error("Failed to get object key")

        if length == 0:
            return String("")

        # Copy via C shim: avoids UnsafePointer-from-Int construction in Mojo.
        # simdjson guarantees valid UTF-8; unsafe_from_utf8 takes raw bytes.
        var bytes = List[UInt8](capacity=length)
        bytes.resize(length, 0)
        self._memcpy_from_addr(Int(bytes.unsafe_ptr()), addr, length)
        return String(unsafe_from_utf8=bytes^)

    def object_iter_get_value(self, iter: Int) -> Int:
        """Get current object value."""
        return self._object_iter_get_value(iter)

    def object_iter_next(self, iter: Int):
        """Move to next object key-value pair."""
        self._object_iter_next(iter)

    def object_iter_free(self, iter: Int):
        """Free object iterator."""
        self._object_iter_free(iter)
