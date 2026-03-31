"""Tests for reflection-based JSON serialization and deserialization."""

from std.testing import assert_equal, assert_true, assert_raises
from std.collections import Optional, List
from mojson import loads, Value, Null
from mojson.reflection import (
    serialize_json,
    serialize_value,
    deserialize_json,
    deserialize_value,
    try_deserialize_json,
    JsonSerializable,
    JsonDeserializable,
)


# ===================================================================
# Test structs
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
struct Product(Defaultable, Movable):
    var name: String
    var price: Float64
    var quantity: Int

    def __init__(out self):
        self.name = ""
        self.price = 0.0
        self.quantity = 0


@fieldwise_init
struct Address(Defaultable, Movable):
    var city: String
    var zip: String

    def __init__(out self):
        self.city = ""
        self.zip = ""


@fieldwise_init
struct Employee(Defaultable, Movable):
    var name: String
    var address: Address

    def __init__(out self):
        self.name = ""
        self.address = Address()


@fieldwise_init
struct Config(Defaultable, Movable):
    var name: String
    var score: Optional[Int]
    var label: Optional[String]

    def __init__(out self):
        self.name = ""
        self.score = None
        self.label = None


@fieldwise_init
struct Stats(Defaultable, Movable):
    var values: List[Int]
    var labels: List[String]

    def __init__(out self):
        self.values = List[Int]()
        self.labels = List[String]()


@fieldwise_init
struct Mixed(Defaultable, Movable):
    var name: String
    var age: Int
    var score: Float64
    var active: Bool
    var tags: List[String]
    var extra: Optional[Int]

    def __init__(out self):
        self.name = ""
        self.age = 0
        self.score = 0.0
        self.active = False
        self.tags = List[String]()
        self.extra = None


struct Empty(Defaultable, Movable):
    def __init__(out self):
        pass


@fieldwise_init
struct WithValue(Defaultable, Movable):
    var name: String
    var metadata: Value

    def __init__(out self):
        self.name = ""
        self.metadata = Value(Null())


# --- Custom trait test structs ---


@fieldwise_init
struct Color(Defaultable, JsonSerializable, Movable):
    """Serialized as "rgb(r,g,b)" string instead of object."""

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
struct RGBString(Defaultable, JsonDeserializable, Movable):
    """Deserialized from a "r,g,b" comma-separated string."""

    var r: Int
    var g: Int
    var b: Int

    def __init__(out self):
        self.r = 0
        self.g = 0
        self.b = 0

    @staticmethod
    def from_json_value(json: Value) raises -> Self:
        # Expects a JSON array like [255, 128, 0]
        if not json.is_array():
            raise Error("RGBString expects a JSON array [r,g,b]")
        var items = json.array_items()
        if len(items) != 3:
            raise Error("RGBString expects exactly 3 elements")
        return Self(
            r=Int(items[0].int_value()),
            g=Int(items[1].int_value()),
            b=Int(items[2].int_value()),
        )


# ===================================================================
# Serialization tests
# ===================================================================


def test_serialize_scalar_struct() raises:
    var p = Point(x=10, y=20)
    var json = serialize_json(p)
    assert_equal(json, '{"x":10,"y":20}')
    print("  test_serialize_scalar_struct passed")


def test_serialize_string_fields() raises:
    var person = Person(name="Alice", age=30, active=True)
    var json = serialize_json(person)
    var parsed = loads(json)
    assert_true(parsed.is_object())
    assert_equal(parsed["name"].string_value(), "Alice")
    assert_equal(Int(parsed["age"].int_value()), 30)
    assert_equal(parsed["active"].bool_value(), True)
    print("  test_serialize_string_fields passed")


def test_serialize_float_fields() raises:
    var prod = Product(name="Widget", price=29.99, quantity=5)
    var json = serialize_json(prod)
    var parsed = loads(json)
    assert_equal(parsed["name"].string_value(), "Widget")
    assert_equal(Int(parsed["quantity"].int_value()), 5)
    print("  test_serialize_float_fields passed")


def test_serialize_nested_struct() raises:
    var emp = Employee(
        name="Bob",
        address=Address(city="NYC", zip="10001"),
    )
    var json = serialize_json(emp)
    var parsed = loads(json)
    assert_equal(parsed["name"].string_value(), "Bob")
    var addr = parsed["address"]
    assert_equal(addr["city"].string_value(), "NYC")
    assert_equal(addr["zip"].string_value(), "10001")
    print("  test_serialize_nested_struct passed")


def test_serialize_optional_present() raises:
    var cfg = Config(name="test", score=42, label=String("ok"))
    var json = serialize_json(cfg)
    var parsed = loads(json)
    assert_equal(Int(parsed["score"].int_value()), 42)
    assert_equal(parsed["label"].string_value(), "ok")
    print("  test_serialize_optional_present passed")


def test_serialize_optional_none() raises:
    var cfg = Config(name="test", score=None, label=None)
    var json = serialize_json(cfg)
    var parsed = loads(json)
    assert_true(parsed["score"].is_null())
    assert_true(parsed["label"].is_null())
    print("  test_serialize_optional_none passed")


def test_serialize_list_int() raises:
    var int_vals = List[Int]()
    int_vals.append(1)
    int_vals.append(2)
    int_vals.append(3)
    var str_lbls = List[String]()
    str_lbls.append("a")
    str_lbls.append("b")
    var s = Stats(values=int_vals^, labels=str_lbls^)
    var json = serialize_json(s)
    var parsed = loads(json)
    var vals = parsed["values"]
    assert_equal(vals.array_count(), 3)
    assert_equal(Int(vals[0].int_value()), 1)
    assert_equal(Int(vals[2].int_value()), 3)
    print("  test_serialize_list_int passed")


def test_serialize_list_string() raises:
    var lbls = List[String]()
    lbls.append("hello")
    lbls.append("world")
    var s = Stats(values=List[Int](), labels=lbls^)
    var json = serialize_json(s)
    var parsed = loads(json)
    var labels = parsed["labels"]
    assert_equal(labels.array_count(), 2)
    assert_equal(labels[0].string_value(), "hello")
    print("  test_serialize_list_string passed")


def test_serialize_empty_struct() raises:
    var e = Empty()
    var json = serialize_json(e)
    assert_equal(json, "{}")
    print("  test_serialize_empty_struct passed")


def test_serialize_pretty() raises:
    var p = Point(x=1, y=2)
    var json = serialize_json[pretty=True](p)
    assert_true("  " in json)
    assert_true("\n" in json)
    print("  test_serialize_pretty passed")


def test_serialize_value_passthrough() raises:
    var raw = loads('{"foo":[1,2,3]}')
    var w = WithValue(name="test", metadata=raw^)
    var json = serialize_json(w)
    var parsed = loads(json)
    assert_equal(parsed["name"].string_value(), "test")
    assert_true(parsed["metadata"].is_object())
    print("  test_serialize_value_passthrough passed")


def test_serialize_string_escaping() raises:
    var p = Person(name='Al "B" C', age=1, active=False)
    var json = serialize_json(p)
    var parsed = loads(json)
    assert_equal(parsed["name"].string_value(), 'Al "B" C')
    print("  test_serialize_string_escaping passed")


# ===================================================================
# Deserialization tests
# ===================================================================


def test_deserialize_scalar_struct() raises:
    var p = deserialize_json[Point]('{"x":10,"y":20}')
    assert_equal(p.x, 10)
    assert_equal(p.y, 20)
    print("  test_deserialize_scalar_struct passed")


def test_deserialize_string_fields() raises:
    var person = deserialize_json[Person](
        '{"name":"Alice","age":30,"active":true}'
    )
    assert_equal(person.name, "Alice")
    assert_equal(person.age, 30)
    assert_equal(person.active, True)
    print("  test_deserialize_string_fields passed")


def test_deserialize_float_fields() raises:
    var prod = deserialize_json[Product](
        '{"name":"Widget","price":29.99,"quantity":5}'
    )
    assert_equal(prod.name, "Widget")
    assert_equal(prod.price, 29.99)
    assert_equal(prod.quantity, 5)
    print("  test_deserialize_float_fields passed")


def test_deserialize_nested_struct() raises:
    var emp = deserialize_json[Employee](
        '{"name":"Bob","address":{"city":"NYC","zip":"10001"}}'
    )
    assert_equal(emp.name, "Bob")
    assert_equal(emp.address.city, "NYC")
    assert_equal(emp.address.zip, "10001")
    print("  test_deserialize_nested_struct passed")


def test_deserialize_optional_present() raises:
    var cfg = deserialize_json[Config](
        '{"name":"test","score":42,"label":"ok"}'
    )
    assert_equal(cfg.name, "test")
    assert_true(Bool(cfg.score))
    assert_equal(cfg.score.value(), 42)
    assert_true(Bool(cfg.label))
    assert_equal(cfg.label.value(), "ok")
    print("  test_deserialize_optional_present passed")


def test_deserialize_optional_missing() raises:
    var cfg = deserialize_json[Config]('{"name":"test"}')
    assert_equal(cfg.name, "test")
    assert_true(not cfg.score)
    assert_true(not cfg.label)
    print("  test_deserialize_optional_missing passed")


def test_deserialize_optional_null() raises:
    var cfg = deserialize_json[Config](
        '{"name":"test","score":null,"label":null}'
    )
    assert_equal(cfg.name, "test")
    assert_true(not cfg.score)
    assert_true(not cfg.label)
    print("  test_deserialize_optional_null passed")


def test_deserialize_list_int() raises:
    var s = deserialize_json[Stats]('{"values":[1,2,3],"labels":["a","b"]}')
    assert_equal(len(s.values), 3)
    assert_equal(s.values[0], 1)
    assert_equal(s.values[1], 2)
    assert_equal(s.values[2], 3)
    assert_equal(len(s.labels), 2)
    assert_equal(s.labels[0], "a")
    print("  test_deserialize_list_int passed")


def test_deserialize_value_passthrough() raises:
    var w = deserialize_json[WithValue](
        '{"name":"test","metadata":{"nested":true}}'
    )
    assert_equal(w.name, "test")
    assert_true(w.metadata.is_object())
    print("  test_deserialize_value_passthrough passed")


# ===================================================================
# Round-trip tests
# ===================================================================


def test_round_trip_point() raises:
    var original = Point(x=42, y=-7)
    var json = serialize_json(original)
    var restored = deserialize_json[Point](json)
    assert_equal(restored.x, original.x)
    assert_equal(restored.y, original.y)
    print("  test_round_trip_point passed")


def test_round_trip_person() raises:
    var original = Person(name="Charlie", age=35, active=True)
    var json = serialize_json(original)
    var restored = deserialize_json[Person](json)
    assert_equal(restored.name, original.name)
    assert_equal(restored.age, original.age)
    assert_equal(restored.active, original.active)
    print("  test_round_trip_person passed")


def test_round_trip_nested() raises:
    var original = Employee(
        name="Dana", address=Address(city="LA", zip="90001")
    )
    var json = serialize_json(original)
    var restored = deserialize_json[Employee](json)
    assert_equal(restored.name, original.name)
    assert_equal(restored.address.city, original.address.city)
    assert_equal(restored.address.zip, original.address.zip)
    print("  test_round_trip_nested passed")


def test_round_trip_mixed() raises:
    var tags = List[String]()
    tags.append("a")
    tags.append("b")
    var original = Mixed(
        name="Eve",
        age=28,
        score=95.5,
        active=True,
        tags=tags^,
        extra=7,
    )
    var json = serialize_json(original)
    var restored = deserialize_json[Mixed](json)
    assert_equal(restored.name, original.name)
    assert_equal(restored.age, original.age)
    assert_equal(restored.score, original.score)
    assert_equal(restored.active, original.active)
    assert_equal(len(restored.tags), 2)
    assert_equal(restored.tags[0], "a")
    assert_true(Bool(restored.extra))
    assert_equal(restored.extra.value(), 7)
    print("  test_round_trip_mixed passed")


# ===================================================================
# Error handling tests
# ===================================================================


def test_error_not_object() raises:
    try:
        var p = deserialize_json[Point]('"not an object"')
        raise Error("Should have raised")
    except e:
        assert_true("Expected JSON object" in String(e))
    print("  test_error_not_object passed")


def test_error_missing_required_field() raises:
    try:
        var p = deserialize_json[Point]('{"x":1}')
        raise Error("Should have raised")
    except e:
        assert_true("not found" in String(e) or "y" in String(e))
    print("  test_error_missing_required_field passed")


def test_error_wrong_type() raises:
    try:
        var p = deserialize_json[Point]('{"x":"nope","y":2}')
        raise Error("Should have raised")
    except e:
        assert_true("not" in String(e) or "int" in String(e))
    print("  test_error_wrong_type passed")


def test_try_deserialize_success() raises:
    var result = try_deserialize_json[Point]('{"x":1,"y":2}')
    assert_true(Bool(result))
    assert_equal(result.value().x, 1)
    assert_equal(result.value().y, 2)
    print("  test_try_deserialize_success passed")


def test_try_deserialize_failure() raises:
    var result = try_deserialize_json[Point]("invalid json!!!")
    assert_true(not result)
    print("  test_try_deserialize_failure passed")


def test_serialize_value_api() raises:
    var p = Point(x=5, y=10)
    var v = serialize_value(p)
    assert_true(v.is_object())
    assert_equal(Int(v["x"].int_value()), 5)
    assert_equal(Int(v["y"].int_value()), 10)
    print("  test_serialize_value_api passed")


def test_deserialize_value_api() raises:
    var json = loads('{"x":3,"y":4}')
    var p = deserialize_value[Point](json)
    assert_equal(p.x, 3)
    assert_equal(p.y, 4)
    print("  test_deserialize_value_api passed")


# ===================================================================
# Custom trait tests
# ===================================================================


def test_custom_serialize() raises:
    """JsonSerializable produces custom JSON instead of reflection."""
    var c = Color(r=255, g=128, b=0)
    var json = serialize_json(c)
    assert_equal(json, '"rgb(255,128,0)"')
    print("  test_custom_serialize passed")


def test_custom_deserialize() raises:
    """JsonDeserializable deserializes from custom JSON array."""
    var json = loads("[255, 128, 0]")
    var rgb = deserialize_value[RGBString](json)
    assert_equal(rgb.r, 255)
    assert_equal(rgb.g, 128)
    assert_equal(rgb.b, 0)
    print("  test_custom_deserialize passed")


# ===================================================================
# GPU round-trip test
# ===================================================================


def test_gpu_roundtrip() raises:
    """Deserialize JSON parsed on GPU into a struct on CPU."""
    var json_str = '{"name":"GPU Test","age":42,"active":true}'
    var person = deserialize_json[Person, target="gpu"](json_str)
    assert_equal(person.name, "GPU Test")
    assert_equal(person.age, 42)
    assert_equal(person.active, True)

    var back = serialize_json(person)
    var rt = deserialize_json[Person, target="gpu"](back)
    assert_equal(rt.name, "GPU Test")
    assert_equal(rt.age, 42)
    print("  test_gpu_roundtrip passed")


# ===================================================================
# Runner
# ===================================================================


def main() raises:
    print("Running reflection-based serde tests...")
    print()

    print("Serialization:")
    test_serialize_scalar_struct()
    test_serialize_string_fields()
    test_serialize_float_fields()
    test_serialize_nested_struct()
    test_serialize_optional_present()
    test_serialize_optional_none()
    test_serialize_list_int()
    test_serialize_list_string()
    test_serialize_empty_struct()
    test_serialize_pretty()
    test_serialize_value_passthrough()
    test_serialize_string_escaping()
    print()

    print("Deserialization:")
    test_deserialize_scalar_struct()
    test_deserialize_string_fields()
    test_deserialize_float_fields()
    test_deserialize_nested_struct()
    test_deserialize_optional_present()
    test_deserialize_optional_missing()
    test_deserialize_optional_null()
    test_deserialize_list_int()
    test_deserialize_value_passthrough()
    print()

    print("Round-trip:")
    test_round_trip_point()
    test_round_trip_person()
    test_round_trip_nested()
    test_round_trip_mixed()
    print()

    print("Error handling:")
    test_error_not_object()
    test_error_missing_required_field()
    test_error_wrong_type()
    test_try_deserialize_success()
    test_try_deserialize_failure()
    test_serialize_value_api()
    test_deserialize_value_api()
    print()

    print("Custom traits:")
    test_custom_serialize()
    test_custom_deserialize()
    print()

    print("GPU round-trip:")
    test_gpu_roundtrip()
    print()

    print("All reflection serde tests passed!")
