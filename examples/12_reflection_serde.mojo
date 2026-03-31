"""Example: Zero-boilerplate struct serde with compile-time reflection.

Demonstrates mojson's reflection-based serialization and deserialization,
which automatically maps struct fields to/from JSON without hand-writing
to_json() or from_json() methods.

Supported field types:
    Scalars: Int, Int64, Bool, Float64, Float32, String
    Containers: List[Int/String/Float64/Bool], Optional[Int/String/Float64/Bool]
    Nested structs, Value (raw JSON pass-through)
"""

from mojson import loads, Value, Null
from mojson import (
    serialize_json,
    serialize_value,
    deserialize_json,
    deserialize_value,
    try_deserialize_json,
    JsonSerializable,
    JsonDeserializable,
)
from std.collections import Optional, List


# ===================================================================
# Struct definitions — only @fieldwise_init needed for serialization.
# Deserialization also requires Defaultable & Movable.
# ===================================================================


@fieldwise_init
struct Point(Defaultable, Movable):
    var x: Int
    var y: Int

    def __init__(out self):
        self.x = 0
        self.y = 0


@fieldwise_init
struct Person(Defaultable, Movable):
    var name: String
    var age: Int
    var active: Bool

    def __init__(out self):
        self.name = ""
        self.age = 0
        self.active = False


@fieldwise_init
struct Address(Defaultable, Movable):
    var city: String
    var zip_code: String

    def __init__(out self):
        self.city = ""
        self.zip_code = ""


@fieldwise_init
struct Employee(Defaultable, Movable):
    var name: String
    var role: String
    var address: Address

    def __init__(out self):
        self.name = ""
        self.role = ""
        self.address = Address()


@fieldwise_init
struct Profile(Defaultable, Movable):
    var username: String
    var bio: Optional[String]
    var score: Optional[Int]
    var tags: List[String]

    def __init__(out self):
        self.username = ""
        self.bio = None
        self.score = None
        self.tags = List[String]()


# --- Custom trait example: serialize as array ---


@fieldwise_init
struct Color(JsonSerializable, Defaultable, Movable):
    """Custom serialization: produces "rgb(r,g,b)" instead of an object."""

    var r: Int
    var g: Int
    var b: Int

    def __init__(out self):
        self.r = 0
        self.g = 0
        self.b = 0

    def to_json_value(self) raises -> Value:
        var s = (
            "rgb("
            + String(self.r)
            + ","
            + String(self.g)
            + ","
            + String(self.b)
            + ")"
        )
        return loads('"' + s + '"')


@fieldwise_init
struct RGBArray(JsonDeserializable, Defaultable, Movable):
    """Custom deserialization: reads from a JSON array [r, g, b]."""

    var r: Int
    var g: Int
    var b: Int

    def __init__(out self):
        self.r = 0
        self.g = 0
        self.b = 0

    @staticmethod
    def from_json_value(json: Value) raises -> Self:
        if not json.is_array():
            raise Error("RGBArray expects a JSON array [r, g, b]")
        var items = json.array_items()
        if len(items) != 3:
            raise Error("RGBArray expects exactly 3 elements")
        return Self(
            r=Int(items[0].int_value()),
            g=Int(items[1].int_value()),
            b=Int(items[2].int_value()),
        )


# ===================================================================
# Examples
# ===================================================================


def example_basic_serialize() raises:
    """Serialize structs with zero boilerplate."""
    print("=== Basic Serialization ===\n")

    var p = Point(x=10, y=20)
    print("Point:", serialize_json(p))

    var person = Person(name="Alice", age=30, active=True)
    print("Person:", serialize_json(person))

    # Pretty-print with 2-space indent
    print("Pretty:\n" + serialize_json[pretty=True](person))


def example_basic_deserialize() raises:
    """Deserialize JSON strings into structs."""
    print("=== Basic Deserialization ===\n")

    var p = deserialize_json[Point]('{"x":42,"y":7}')
    print("Point: x=" + String(p.x) + " y=" + String(p.y))

    var person = deserialize_json[Person](
        '{"name":"Bob","age":25,"active":false}'
    )
    print(
        "Person: "
        + person.name
        + " age="
        + String(person.age)
        + " active="
        + ("true" if person.active else "false")
    )
    print()


def example_nested_structs() raises:
    """Nested structs are handled recursively."""
    print("=== Nested Structs ===\n")

    var emp = Employee(
        name="Carol",
        role="Engineer",
        address=Address(city="San Francisco", zip_code="94105"),
    )
    var json = serialize_json(emp)
    print("Serialized:", json)

    var restored = deserialize_json[Employee](json)
    print(
        "Restored: "
        + restored.name
        + " in "
        + restored.address.city
        + " ("
        + restored.address.zip_code
        + ")"
    )
    print()


def example_optional_and_list() raises:
    """Optional fields and List fields."""
    print("=== Optional & List Fields ===\n")

    var tags = List[String]()
    tags.append("mojo")
    tags.append("json")
    var profile = Profile(
        username="dev42",
        bio=String("Mojo enthusiast"),
        score=Int(100),
        tags=tags^,
    )
    print("With values:", serialize_json(profile))

    # Missing optional fields default to None
    var minimal = deserialize_json[Profile](
        '{"username":"anon","tags":[]}'
    )
    print(
        "Minimal: username="
        + minimal.username
        + " bio="
        + ("None" if not minimal.bio else minimal.bio.value())
    )
    print()


def example_round_trip() raises:
    """Serialize then deserialize — data survives the round trip."""
    print("=== Round-Trip ===\n")

    var original = Person(name="Dave", age=40, active=True)
    var json = serialize_json(original)
    var restored = deserialize_json[Person](json)

    print("Original: " + original.name + " age=" + String(original.age))
    print("JSON:     " + json)
    print("Restored: " + restored.name + " age=" + String(restored.age))

    var names_match = original.name == restored.name
    var ages_match = original.age == restored.age
    print("Match:    " + ("true" if names_match and ages_match else "false"))
    print()


def example_gpu_parsing() raises:
    """Parse JSON on GPU, deserialize struct on CPU."""
    print("=== GPU Round-Trip ===\n")

    var json_str = '{"name":"GPU User","age":99,"active":true}'
    var person = deserialize_json[Person, target="gpu"](json_str)
    print(
        "Deserialized (GPU): "
        + person.name
        + " age="
        + String(person.age)
    )
    print()


def example_try_deserialize() raises:
    """Non-raising deserialization returns Optional."""
    print("=== try_deserialize_json ===\n")

    var good = try_deserialize_json[Point]('{"x":1,"y":2}')
    if good:
        print(
            "Success: x="
            + String(good.value().x)
            + " y="
            + String(good.value().y)
        )
    else:
        print("Failed (unexpected)")

    var bad = try_deserialize_json[Point]("not json")
    if bad:
        print("Success (unexpected)")
    else:
        print("Graceful failure on invalid JSON: None")
    print()


def example_value_api() raises:
    """Work with Value objects directly."""
    print("=== Value API ===\n")

    var person = Person(name="Eve", age=28, active=True)
    var val = serialize_value(person)
    print("Value type: " + ("object" if val.is_object() else "other"))
    print("Name field: " + val["name"].string_value())

    var back = deserialize_value[Person](val)
    print("Back to struct: " + back.name)
    print()


def example_custom_traits() raises:
    """Override reflection with JsonSerializable / JsonDeserializable."""
    print("=== Custom Traits ===\n")

    # JsonSerializable — Color serializes as "rgb(r,g,b)"
    var c = Color(r=255, g=128, b=0)
    print("Color JSON: " + serialize_json(c))

    # JsonDeserializable — RGBArray deserializes from [r, g, b]
    var json = loads("[100, 200, 50]")
    var rgb = deserialize_value[RGBArray](json)
    print(
        "RGBArray: r="
        + String(rgb.r)
        + " g="
        + String(rgb.g)
        + " b="
        + String(rgb.b)
    )
    print()


def example_error_messages() raises:
    """Rich error messages tell you exactly what went wrong."""
    print("=== Error Messages ===\n")

    try:
        _ = deserialize_json[Point]("42")
    except e:
        print("Not an object: " + String(e))

    try:
        _ = deserialize_json[Person]('{"name":"Alice"}')
    except e:
        print("Missing field: " + String(e))

    try:
        _ = deserialize_json[Point]('{"x":"oops","y":1}')
    except e:
        print("Wrong type: " + String(e))
    print()


def main() raises:
    print("\n" + "=" * 50)
    print("  Reflection-Based Serde (Zero Boilerplate)")
    print("=" * 50 + "\n")

    example_basic_serialize()
    example_basic_deserialize()
    example_nested_structs()
    example_optional_and_list()
    example_round_trip()
    example_gpu_parsing()
    example_try_deserialize()
    example_value_api()
    example_custom_traits()
    example_error_messages()

    print("=" * 50)
    print("  All examples completed successfully!")
    print("=" * 50)
