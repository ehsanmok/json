# Working with the Value type
#
# Demonstrates: Value type checking and value extraction

from json import loads, dumps, Value, Null
from std.collections import Dict, List


def main() raises:
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

    # Integers above Int64.MAX get their own type, readable with
    # uint_value() and constructible with Value(UInt64).
    var big_val = Value(UInt64(18446744073709551615))
    print("  UInt:", dumps(big_val), "is_uint:", big_val.is_uint())

    # Containers can be built in one statement from a List or a Dict.
    var list_val = Value([Value(1), Value(2), Value(3)])
    var members = Dict[String, Value]()
    members["scores"] = list_val.copy()
    var obj_val = Value(members^)
    print("  Array:", dumps(list_val))
    print("  Object:", dumps(obj_val))
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

    var data = loads(
        '{"name": "Alice", "age": 30, "score": 95.5, "active": true}'
    )

    # The *_value() readers never raise and never check the tag: they
    # read the slot they are told to, so int_value() on a string is 0.
    print("  String value:", data["name"].string_value())
    print("  Int value:", data["age"].int_value())
    print("  Float value:", data["score"].float_value())
    print("  Bool value:", data["active"].bool_value())

    # The as_*() readers check first and raise an error naming the type
    # they actually found, which is what you want when the document
    # came from somewhere you do not control.
    try:
        _ = data["name"].as_int()
    except e:
        print("  as_int on a string:", e)

    # And the *_or() readers substitute a default instead of raising.
    print("  int_or on a string:", data["name"].int_or(-1))
    print()

    # Dict-style lookup: get() answers None rather than raising, and
    # the two-argument form substitutes a fallback.
    print("Lookups:")
    var missing = data.get("nickname")
    print("  get('nickname') found something:", Bool(missing))
    print("  get('nickname', default):", data.get("nickname", Value("none")))
    print("  'age' in data:", "age" in data)
    print("  try_at('/nope'):", Bool(data.try_at("/nope")))
    print()

    # Array and object metadata
    print("Array/Object metadata:")

    var arr = loads("[10, 20, 30, 40, 50]")
    print("  Array count:", arr.array_count(), "len:", len(arr))
    print("  Array raw JSON:", arr.raw_json())
    print("  Last element:", arr[-1])
    print("  30 in arr:", 30 in arr)

    var obj = loads('{"a": 1, "b": 2, "c": 3}')
    print("  Object count:", obj.object_count(), "len:", len(obj))
    print("  Object raw JSON:", obj.raw_json())
    var obj_keys = obj.object_keys()
    print("  Object keys:", len(obj_keys), "keys")
    print()

    # Iteration: lazy, one element or member at a time. array_items()
    # and object_items() build the whole list first, so prefer these.
    print("Iteration:")
    var total = Int64(0)
    for item in arr:
        total += item.int_value()
    print("  Array sum:", total)
    for pair in obj.items():
        print("  Member:", pair[0], "=", pair[1])
    print()

    # Value equality
    print("Value equality:")
    var v1 = loads("42")
    var v2 = loads("42")
    var v3 = loads("43")
    print("  42 == 42:", v1 == v2)
    print("  42 == 43:", v1 == v3)
    print("  42 != 43:", v1 != v3)

    # Equality is structural, not textual: an object's members carry no
    # order, and JSON has a single number type.
    print("  {a,b} == {b,a}:", loads('{"a":1,"b":2}') == loads('{"b":2,"a":1}'))
    print("  1 == 1.0:", Value(1) == Value(1.0))

    # Which is also what lets a Value key a Dict.
    var index = Dict[Value, String]()
    index[loads('{"a":1,"b":2}')] = "found"
    print("  Dict lookup by a reordered key:", index[loads('{"b":2,"a":1}')])
