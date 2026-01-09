# Example 01: Basic JSON Parsing
#
# Demonstrates: loads() and dumps() for string-based JSON operations

from src import loads, dumps, Value


fn main() raises:
    # Parse a simple JSON object
    var json_str = '{"name": "Alice", "age": 30, "active": true}'
    var data = loads(json_str)

    print("Parsed JSON:")
    print(dumps(data))
    print()

    # Parse different JSON types
    print("Parsing different value types:")

    # Integer
    var int_val = loads("42")
    print("  Integer:", dumps(int_val))

    # Float
    var float_val = loads("3.14159")
    print("  Float:", dumps(float_val))

    # String
    var str_val = loads('"Hello, World!"')
    print("  String:", dumps(str_val))

    # Boolean
    var bool_val = loads("true")
    print("  Boolean:", dumps(bool_val))

    # Null
    var null_val = loads("null")
    print("  Null:", dumps(null_val))

    # Array
    var arr_val = loads('[1, 2, 3, "four", true, null]')
    print("  Array:", dumps(arr_val))

    # Nested object
    var nested = loads(
        '{"user": {"name": "Bob", "scores": [95, 87, 92]}, "timestamp": 1234567890}'
    )
    print("  Nested:", dumps(nested))
    print()

    # Roundtrip: parse and re-serialize should give equivalent JSON
    var original = '{"key":"value","count":100}'
    var parsed = loads(original)
    var serialized = dumps(parsed)
    print("Roundtrip test:")
    print("  Original:  ", original)
    print("  Serialized:", serialized)
