# Example 03: Working with Value Types
#
# Demonstrates: Value type checking and value extraction

from src import loads, dumps, Value, Null, make_array_value, make_object_value
from collections import List


fn main() raises:
    # Create Values directly (not from parsing)
    print("Creating Values directly:")

    var null_val = Value(Null())
    print("  Null:", dumps(null_val))

    var bool_val = Value(True)
    print("  Bool:", dumps(bool_val))

    var int_val = Value(42)
    print("  Int:", dumps(int_val))

    var float_val = Value(3.14159)
    print("  Float:", dumps(float_val))

    var str_val = Value("Hello, Mojo!")
    print("  String:", dumps(str_val))
    print()

    # Type checking
    print("Type checking parsed values:")

    var parsed_null = loads("null")
    var parsed_bool = loads("true")
    var parsed_int = loads("123")
    var parsed_float = loads("45.67")
    var parsed_string = loads('"text"')
    var parsed_array = loads("[1, 2, 3]")
    var parsed_object = loads('{"key": "value"}')

    print("  null is_null:", parsed_null.is_null())
    print("  true is_bool:", parsed_bool.is_bool())
    print("  123 is_int:", parsed_int.is_int())
    print("  45.67 is_float:", parsed_float.is_float())
    print("  'text' is_string:", parsed_string.is_string())
    print("  [1,2,3] is_array:", parsed_array.is_array())
    print("  {...} is_object:", parsed_object.is_object())
    print()

    # is_number() returns True for both int and float
    print("Number checking:")
    print("  123 is_number:", parsed_int.is_number())
    print("  45.67 is_number:", parsed_float.is_number())
    print("  'text' is_number:", parsed_string.is_number())
    print()

    # Value extraction
    print("Extracting values:")

    var data = loads('{"name": "Alice", "age": 30, "score": 95.5, "active": true}')
    # Note: The current API stores arrays/objects as raw JSON strings
    # To extract individual fields, you would parse them separately

    var name = loads('"Alice"')
    var age = loads("30")
    var score = loads("95.5")
    var active = loads("true")

    print("  String value:", name.string_value())
    print("  Int value:", age.int_value())
    print("  Float value:", score.float_value())
    print("  Bool value:", active.bool_value())
    print()

    # Array and object metadata
    print("Array/Object metadata:")

    var arr = loads("[10, 20, 30, 40, 50]")
    print("  Array count:", arr.array_count())
    print("  Array raw JSON:", arr.raw_json())

    var obj = loads('{"a": 1, "b": 2, "c": 3}')
    print("  Object count:", obj.object_count())
    print("  Object raw JSON:", obj.raw_json())
    print("  Object keys:", obj.object_keys().__str__())
    print()

    # Value equality
    print("Value equality:")
    var v1 = loads("42")
    var v2 = loads("42")
    var v3 = loads("43")
    print("  42 == 42:", v1 == v2)
    print("  42 == 43:", v1 == v3)
    print("  42 != 43:", v1 != v3)
