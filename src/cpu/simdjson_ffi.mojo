# simdjson FFI wrapper for Mojo
# Provides high-performance JSON parsing via simdjson C++ library
# Uses OwnedDLHandle for runtime library loading

from sys.ffi import OwnedDLHandle, external_call
from memory import UnsafePointer, Span
from collections import List

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
    var _create_parser: fn () -> Int
    var _destroy_parser: fn (Int) -> None
    var _parse: fn (Int, Int, Int) -> Int
    var _get_root: fn (Int) -> Int

    # Value functions
    var _value_get_type: fn (Int) -> Int
    var _value_get_bool: fn (Int, Int) -> Int
    var _value_get_int64: fn (Int, Int) -> Int
    var _value_get_uint64: fn (Int, Int) -> Int
    var _value_get_double: fn (Int, Int) -> Int
    var _value_get_string: fn (Int, Int, Int) -> Int
    var _value_free: fn (Int) -> None

    # Array functions
    var _array_begin: fn (Int) -> Int
    var _array_iter_done: fn (Int) -> Int
    var _array_iter_get: fn (Int) -> Int
    var _array_iter_next: fn (Int) -> None
    var _array_iter_free: fn (Int) -> None
    var _array_count: fn (Int) -> Int

    # Object functions
    var _object_begin: fn (Int) -> Int
    var _object_iter_done: fn (Int) -> Int
    var _object_iter_get_key: fn (Int, Int, Int) -> None
    var _object_iter_get_value: fn (Int) -> Int
    var _object_iter_next: fn (Int) -> None
    var _object_iter_free: fn (Int) -> None
    var _object_count: fn (Int) -> Int

    fn __init__(
        out self, lib_path: String = "build/libsimdjson_wrapper.so"
    ) raises:
        """Initialize by loading the simdjson wrapper library."""
        self._lib = OwnedDLHandle(lib_path)

        # Parser functions
        self._create_parser = self._lib.get_function[fn () -> Int](
            "simdjson_create_parser"
        )
        self._destroy_parser = self._lib.get_function[fn (Int) -> None](
            "simdjson_destroy_parser"
        )
        self._parse = self._lib.get_function[fn (Int, Int, Int) -> Int](
            "simdjson_parse"
        )
        self._get_root = self._lib.get_function[fn (Int) -> Int](
            "simdjson_get_root"
        )

        # Value functions
        self._value_get_type = self._lib.get_function[fn (Int) -> Int](
            "simdjson_value_get_type"
        )
        self._value_get_bool = self._lib.get_function[fn (Int, Int) -> Int](
            "simdjson_value_get_bool"
        )
        self._value_get_int64 = self._lib.get_function[fn (Int, Int) -> Int](
            "simdjson_value_get_int64"
        )
        self._value_get_uint64 = self._lib.get_function[fn (Int, Int) -> Int](
            "simdjson_value_get_uint64"
        )
        self._value_get_double = self._lib.get_function[fn (Int, Int) -> Int](
            "simdjson_value_get_double"
        )
        self._value_get_string = self._lib.get_function[
            fn (Int, Int, Int) -> Int
        ]("simdjson_value_get_string")
        self._value_free = self._lib.get_function[fn (Int) -> None](
            "simdjson_value_free"
        )

        # Array functions
        self._array_begin = self._lib.get_function[fn (Int) -> Int](
            "simdjson_array_begin"
        )
        self._array_iter_done = self._lib.get_function[fn (Int) -> Int](
            "simdjson_array_iter_done"
        )
        self._array_iter_get = self._lib.get_function[fn (Int) -> Int](
            "simdjson_array_iter_get"
        )
        self._array_iter_next = self._lib.get_function[fn (Int) -> None](
            "simdjson_array_iter_next"
        )
        self._array_iter_free = self._lib.get_function[fn (Int) -> None](
            "simdjson_array_iter_free"
        )
        self._array_count = self._lib.get_function[fn (Int) -> Int](
            "simdjson_array_count"
        )

        # Object functions
        self._object_begin = self._lib.get_function[fn (Int) -> Int](
            "simdjson_object_begin"
        )
        self._object_iter_done = self._lib.get_function[fn (Int) -> Int](
            "simdjson_object_iter_done"
        )
        self._object_iter_get_key = self._lib.get_function[
            fn (Int, Int, Int) -> None
        ]("simdjson_object_iter_get_key")
        self._object_iter_get_value = self._lib.get_function[fn (Int) -> Int](
            "simdjson_object_iter_get_value"
        )
        self._object_iter_next = self._lib.get_function[fn (Int) -> None](
            "simdjson_object_iter_next"
        )
        self._object_iter_free = self._lib.get_function[fn (Int) -> None](
            "simdjson_object_iter_free"
        )
        self._object_count = self._lib.get_function[fn (Int) -> Int](
            "simdjson_object_count"
        )

        # Create the parser
        self._parser = self._create_parser()
        if self._parser == 0:
            raise Error("Failed to create simdjson parser")

    fn destroy(mut self):
        """Clean up the parser. Call this explicitly when done."""
        if self._parser != 0:
            self._destroy_parser(self._parser)
            self._parser = 0

    fn parse(mut self, json: String) raises -> Int:
        """Parse JSON and return root value handle."""
        var json_copy = json
        var c_str = json_copy.as_c_string_slice()
        var ptr = Int(c_str.unsafe_ptr())
        var length = len(json_copy)

        var err = self._parse(self._parser, ptr, length)

        if err != SIMDJSON_OK:
            if err == SIMDJSON_ERROR_INVALID_JSON:
                raise Error("Invalid JSON")
            elif err == SIMDJSON_ERROR_UTF8:
                raise Error("Invalid UTF-8 in JSON")
            elif err == SIMDJSON_ERROR_CAPACITY:
                raise Error("JSON too large")
            else:
                raise Error("Unknown simdjson error")

        return self._get_root(self._parser)

    fn get_type(self, value: Int) -> Int:
        """Get the type of a value."""
        return self._value_get_type(value)

    fn get_bool(self, value: Int) raises -> Bool:
        """Get value as boolean."""
        var result = List[Int32](capacity=1)
        result.append(0)
        var err = self._value_get_bool(value, Int(result.unsafe_ptr()))
        if err != SIMDJSON_OK:
            raise Error("Value is not a boolean")
        return result[0] != 0

    fn get_int(self, value: Int) raises -> Int64:
        """Get value as int64."""
        var result = List[Int64](capacity=1)
        result.append(0)
        var err = self._value_get_int64(value, Int(result.unsafe_ptr()))
        if err != SIMDJSON_OK:
            raise Error("Value is not an integer")
        return result[0]

    fn get_uint(self, value: Int) raises -> UInt64:
        """Get value as uint64."""
        var result = List[UInt64](capacity=1)
        result.append(0)
        var err = self._value_get_uint64(value, Int(result.unsafe_ptr()))
        if err != SIMDJSON_OK:
            raise Error("Value is not an unsigned integer")
        return result[0]

    fn get_float(self, value: Int) raises -> Float64:
        """Get value as double."""
        var result = List[Float64](capacity=1)
        result.append(0.0)
        var err = self._value_get_double(value, Int(result.unsafe_ptr()))
        if err != SIMDJSON_OK:
            raise Error("Value is not a float")
        return result[0]

    fn get_string(self, value: Int) raises -> String:
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

        # Use external_call memcpy then unsafe_from_utf8 - simdjson guarantees valid UTF-8
        # Don't include null terminator - unsafe_from_utf8 takes raw bytes as the string content
        var bytes = List[UInt8](capacity=length)
        bytes.resize(length, 0)
        external_call["memcpy", NoneType](Int(bytes.unsafe_ptr()), addr, length)
        return String(unsafe_from_utf8=bytes^)

    fn free_value(self, value: Int):
        """Free a value handle."""
        self._value_free(value)

    fn array_count(self, value: Int) -> Int:
        """Get array element count."""
        return self._array_count(value)

    fn array_begin(self, value: Int) -> Int:
        """Start iterating over array."""
        return self._array_begin(value)

    fn array_iter_done(self, iter: Int) -> Bool:
        """Check if array iteration is done."""
        return self._array_iter_done(iter) != 0

    fn array_iter_get(self, iter: Int) -> Int:
        """Get current array element."""
        return self._array_iter_get(iter)

    fn array_iter_next(self, iter: Int):
        """Move to next array element."""
        self._array_iter_next(iter)

    fn array_iter_free(self, iter: Int):
        """Free array iterator."""
        self._array_iter_free(iter)

    fn object_count(self, value: Int) -> Int:
        """Get object key count."""
        return self._object_count(value)

    fn object_begin(self, value: Int) -> Int:
        """Start iterating over object."""
        return self._object_begin(value)

    fn object_iter_done(self, iter: Int) -> Bool:
        """Check if object iteration is done."""
        return self._object_iter_done(iter) != 0

    fn object_iter_get_key(self, iter: Int) raises -> String:
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

        # Use external_call memcpy then unsafe_from_utf8 - simdjson guarantees valid UTF-8
        # Don't include null terminator - unsafe_from_utf8 takes raw bytes as the string content
        var bytes = List[UInt8](capacity=length)
        bytes.resize(length, 0)
        external_call["memcpy", NoneType](Int(bytes.unsafe_ptr()), addr, length)
        return String(unsafe_from_utf8=bytes^)

    fn object_iter_get_value(self, iter: Int) -> Int:
        """Get current object value."""
        return self._object_iter_get_value(iter)

    fn object_iter_next(self, iter: Int):
        """Move to next object key-value pair."""
        self._object_iter_next(iter)

    fn object_iter_free(self, iter: Int):
        """Free object iterator."""
        self._object_iter_free(iter)
