"""Tests for reflection-based JSON serialization and deserialization."""

from std.testing import assert_equal, assert_true, assert_raises
from std.collections import Optional, List, Dict
from json import loads, Value, Null
from json.reflection import (
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
struct SizedInts(Defaultable, Movable):
    """Every sized integer width, to pin the reflection arms."""

    var i8: Int8
    var i16: Int16
    var i32: Int32
    var i64: Int64
    var u8: UInt8
    var u16: UInt16
    var u32: UInt32
    var u64: UInt64

    def __init__(out self):
        self.i8 = 0
        self.i16 = 0
        self.i32 = 0
        self.i64 = 0
        self.u8 = 0
        self.u16 = 0
        self.u32 = 0
        self.u64 = 0


@fieldwise_init
struct LineItem(Movable):
    var sku: String
    var qty: Int


@fieldwise_init
struct OrderLine(Defaultable, Movable):
    var code: String
    var qty: Int64

    def __init__(out self):
        self.code = ""
        self.qty = 0


@fieldwise_init
struct Order(Defaultable, Movable):
    var order_id: String
    var lines: List[OrderLine]

    def __init__(out self):
        self.order_id = ""
        self.lines = List[OrderLine]()


@fieldwise_init
struct Basket(Movable):
    """A list of structs, and a type with no default constructor.

    Both were refused by the old read path; neither is refused now.
    """

    var items: List[LineItem]


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
    """A struct needs an object, and the error says which brace it wanted."""
    try:
        var p = deserialize_json[Point]('"not an object"')
        raise Error("Should have raised")
    except e:
        assert_true("expected '{'" in String(e), String(e))
    print("  test_error_not_object passed")


def test_error_missing_required_field() raises:
    """A missing field names itself rather than leaving a half-built struct."""
    try:
        var p = deserialize_json[Point]('{"x":1}')
        raise Error("Should have raised")
    except e:
        assert_true("missing required field 'y'" in String(e), String(e))
    print("  test_error_missing_required_field passed")


def test_error_wrong_type() raises:
    """A type mismatch names the field it happened in."""
    try:
        var p = deserialize_json[Point]('{"x":"nope","y":2}')
        raise Error("Should have raised")
    except e:
        assert_true("field 'x'" in String(e), String(e))
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
# Combinator types: Dict[String, T], nested lists, Optional<->List.
# ===================================================================


@fieldwise_init
struct CombinatorBox(Defaultable, Movable):
    var counts: Dict[String, Int]
    var labels: Dict[String, String]
    var tags: List[Optional[String]]
    var maybe_ids: Optional[List[Int]]
    var matrix: List[List[Int]]

    def __init__(out self):
        self.counts = Dict[String, Int]()
        self.labels = Dict[String, String]()
        self.tags = List[Optional[String]]()
        self.maybe_ids = None
        self.matrix = List[List[Int]]()


def test_serialize_dict_string_int() raises:
    var d = Dict[String, Int]()
    d["a"] = 1
    d["b"] = 2
    var json = serialize_json(d)
    var parsed = loads(json)
    assert_true(parsed.is_object())
    assert_equal(parsed["a"].int_value(), 1)
    assert_equal(parsed["b"].int_value(), 2)
    print("  test_serialize_dict_string_int passed")


def test_serialize_dict_string_string() raises:
    var d = Dict[String, String]()
    d["en"] = String("hello")
    d["es"] = String("hola")
    var json = serialize_json(d)
    var parsed = loads(json)
    assert_equal(parsed["en"].string_value(), String("hello"))
    assert_equal(parsed["es"].string_value(), String("hola"))
    print("  test_serialize_dict_string_string passed")


def test_serialize_list_optional_int() raises:
    var lst = List[Optional[Int]]()
    lst.append(Int(1))
    lst.append(None)
    lst.append(Int(3))
    var json = serialize_json(lst)
    assert_equal(json, "[1,null,3]")
    print("  test_serialize_list_optional_int passed")


def test_serialize_optional_list_int_present() raises:
    var lst = List[Int]()
    lst.append(1)
    lst.append(2)
    lst.append(3)
    var opt: Optional[List[Int]] = lst^
    var json = serialize_json(opt)
    assert_equal(json, "[1,2,3]")
    print("  test_serialize_optional_list_int_present passed")


def test_serialize_optional_list_int_none() raises:
    var opt = Optional[List[Int]](None)
    var json = serialize_json(opt)
    assert_equal(json, "null")
    print("  test_serialize_optional_list_int_none passed")


def test_serialize_list_list_int() raises:
    var inner_a = List[Int]()
    inner_a.append(1)
    inner_a.append(2)
    var inner_b = List[Int]()
    inner_b.append(3)
    var outer = List[List[Int]]()
    outer.append(inner_a^)
    outer.append(inner_b^)
    var json = serialize_json(outer)
    assert_equal(json, "[[1,2],[3]]")
    print("  test_serialize_list_list_int passed")


def test_round_trip_combinator_box() raises:
    var box = CombinatorBox()
    box.counts["a"] = 1
    box.counts["b"] = 2
    box.labels["k"] = String("v")

    box.tags = List[Optional[String]]()
    box.tags.append(String("x"))
    box.tags.append(None)
    box.tags.append(String("y"))

    var ids = List[Int]()
    ids.append(7)
    ids.append(8)
    box.maybe_ids = ids^

    var row1 = List[Int]()
    row1.append(1)
    row1.append(2)
    var row2 = List[Int]()
    row2.append(3)
    box.matrix = List[List[Int]]()
    box.matrix.append(row1^)
    box.matrix.append(row2^)

    var json = serialize_json(box)
    var back = deserialize_json[CombinatorBox](json)

    assert_equal(back.counts["a"], 1)
    assert_equal(back.counts["b"], 2)
    assert_equal(back.labels["k"], String("v"))
    assert_equal(len(back.tags), 3)
    assert_equal(back.tags[0].value(), String("x"))
    assert_true(not back.tags[1])
    assert_equal(back.tags[2].value(), String("y"))
    assert_true(back.maybe_ids.__bool__())
    assert_equal(back.maybe_ids.value()[0], 7)
    assert_equal(back.maybe_ids.value()[1], 8)
    assert_equal(len(back.matrix), 2)
    assert_equal(back.matrix[0][0], 1)
    assert_equal(back.matrix[0][1], 2)
    assert_equal(back.matrix[1][0], 3)
    print("  test_round_trip_combinator_box passed")


def test_round_trip_combinator_box_null_optional_list() raises:
    var box = CombinatorBox()
    box.maybe_ids = None
    var json = serialize_json(box)
    var back = deserialize_json[CombinatorBox](json)
    assert_true(not back.maybe_ids)
    print("  test_round_trip_combinator_box_null_optional_list passed")


# ===================================================================
# Runner
# ===================================================================


# ---------------------------------------------------------------------------
# Sized integer widths.
#
# Before these arms existed, only `Int` and `Int64` were reflected; an
# `Int32` field fell through to the "Unsupported field type" error, which
# is what forced a caller with an `Int32` field off the reflection path
# entirely.
# ---------------------------------------------------------------------------


def test_serialize_sized_ints() raises:
    var v = SizedInts(-8, -16, -32, -64, 8, 16, 32, 64)
    var json = serialize_json(v)
    assert_equal(
        json,
        '{"i8":-8,"i16":-16,"i32":-32,"i64":-64,'
        + '"u8":8,"u16":16,"u32":32,"u64":64}',
    )
    print("  test_serialize_sized_ints passed")


def test_deserialize_sized_ints() raises:
    var json = (
        '{"i8":-1,"i16":-2,"i32":-3,"i64":-4,'
        + '"u8":1,"u16":2,"u32":3,"u64":4}'
    )
    var v = deserialize_json[SizedInts](json)
    assert_equal(Int(v.i8), -1)
    assert_equal(Int(v.i16), -2)
    assert_equal(Int(v.i32), -3)
    assert_equal(Int(v.i64), -4)
    assert_equal(Int(v.u8), 1)
    assert_equal(Int(v.u16), 2)
    assert_equal(Int(v.u32), 3)
    assert_equal(Int(v.u64), 4)
    print("  test_deserialize_sized_ints passed")


def test_round_trip_sized_ints() raises:
    var original = SizedInts(-128, -32768, -2147483648, -64, 255, 65535, 7, 9)
    var back = deserialize_json[SizedInts](serialize_json(original))
    assert_equal(Int(back.i8), -128)
    assert_equal(Int(back.i16), -32768)
    assert_equal(Int(back.i32), -2147483648)
    assert_equal(Int(back.u8), 255)
    assert_equal(Int(back.u16), 65535)
    print("  test_round_trip_sized_ints passed")


def test_serialize_int32_negative() raises:
    """The specific gap that pushed callers off reflection."""
    var v = SizedInts()
    v.i32 = -12345
    assert_true('"i32":-12345' in serialize_json(v))
    print("  test_serialize_int32_negative passed")


# ---------------------------------------------------------------------------
# Unsupported list element types must fail loudly.
#
# A `List` is itself a struct, so `List[LineItem]` used to reach the
# nested-struct arm: serialization reflected over List's *internal*
# fields and emitted its data pointer, length and capacity as a JSON
# object -- corrupt output that looked like success. Both directions now
# raise instead.
# ---------------------------------------------------------------------------


@fieldwise_init
struct FloatLists(Defaultable, Movable):
    """Float containers, which a type-name substring test used to break."""

    var samples: List[Float64]
    var ratio: Float32
    var reading: Float64

    def __init__(out self):
        self.samples = List[Float64]()
        self.ratio = 0.0
        self.reading = 0.0


def test_serialize_list_of_float64() raises:
    """A `List[Float64]` field is an array, not a float.

    Float dispatch matched the reflected type *name* against the
    substring "SIMD[DType.float64". `List[Float64]` reflects to a name
    containing its element type, so the list matched the float arm and
    `rebind[Float64]` on a list's (pointer, length, capacity) layout
    failed to compile -- any struct with such a field was unusable, and
    the error pointed at the rebind rather than at the field. Dispatch
    now compares types (`T == Float64`), which no container can alias.
    """
    var v = FloatLists(List[Float64](), Float32(0.5), 1.25)
    v.samples.append(1.5)
    v.samples.append(2.0)
    v.samples.append(-0.25)
    var json = serialize_json(v)
    assert_equal(json, '{"samples":[1.5,2.0,-0.25],"ratio":0.5,"reading":1.25}')
    print("  test_serialize_list_of_float64 passed")


def test_round_trip_list_of_float64() raises:
    """The same shape survives a round trip through the reader."""
    var v = FloatLists(List[Float64](), Float32(0.25), 3.5)
    v.samples.append(0.125)
    v.samples.append(64.0)
    var back = deserialize_json[FloatLists](serialize_json(v))
    assert_equal(len(back.samples), 2)
    assert_equal(back.samples[0], 0.125)
    assert_equal(back.samples[1], 64.0)
    assert_equal(back.ratio, Float32(0.25))
    assert_equal(back.reading, 3.5)
    print("  test_round_trip_list_of_float64 passed")


def test_serialize_list_of_structs() raises:
    """`List[<struct>]` serializes as an array of objects.

    This used to be impossible two different ways. Originally a `List`
    is itself a struct, so it reached the nested-struct arm and
    reflection emitted List's internal data pointer, length and capacity
    as a JSON object -- corrupt output that looked like success. It was
    then made an explicit error, on the belief that the element type
    could not be recovered: inside a function parametric on `T` a
    reflected field type is bound only by `AnyType`, so no `[E](List[E])`
    overload matches and even `len()` will not resolve.

    Retroactive conformance is what makes it work. Inside
    `__extension List(_JsonEmit)` the element parameter is concrete per
    instantiation, so an ordinary generic call deduces it.
    """
    var b = Basket(List[LineItem]())
    b.items.append(LineItem("sku-1", 1))
    b.items.append(LineItem("sku-2", 2))
    var json = serialize_json(b)
    assert_equal(
        json,
        '{"items":[{"sku":"sku-1","qty":1},{"sku":"sku-2","qty":2}]}',
    )
    print("  test_serialize_list_of_structs passed")


def test_serialize_empty_list_of_structs() raises:
    var b = Basket(List[LineItem]())
    assert_equal(serialize_json(b), '{"items":[]}')
    print("  test_serialize_empty_list_of_structs passed")


def test_serialize_list_of_structs_nested_deeper() raises:
    """A struct in a list in a struct, with scalars alongside."""
    var o = Order()
    o.order_id = "ord-9"
    o.lines.append(OrderLine("a", 2))
    o.lines.append(OrderLine("b", 3))
    assert_equal(
        serialize_json(o),
        '{"order_id":"ord-9","lines":'
        + '[{"code":"a","qty":2},{"code":"b","qty":3}]}',
    )
    print("  test_serialize_list_of_structs_nested_deeper passed")


def test_deserialize_list_of_structs() raises:
    """A list of structs decodes, rather than being declined.

    The read path used to refuse this outright: filling a `List[E]`
    through a reflected field pointer needed `E` to be
    default-constructible, so it could not be done at all. Reading
    into a slot rather than over a default removes the requirement.
    """
    var basket = deserialize_json[Basket](
        '{"items":[{"sku":"a","qty":2},{"sku":"b","qty":3}]}'
    )
    assert_equal(len(basket.items), 2)
    assert_equal(basket.items[0].sku, "a")
    assert_equal(basket.items[0].qty, 2)
    assert_equal(basket.items[1].sku, "b")
    assert_equal(basket.items[1].qty, 3)

    var empty = deserialize_json[Basket]('{"items":[]}')
    assert_equal(len(empty.items), 0)
    print("  test_deserialize_list_of_structs passed")


def test_error_inside_a_list_names_the_element() raises:
    """A failure deep in a container reports the path to it."""
    var message = String()
    try:
        _ = deserialize_json[Basket]('{"items":[{"sku":"a","qty":"no"}]}')
    except e:
        message = String(e)
    assert_true("field 'items'" in message, message)
    assert_true("element 0" in message, message)
    assert_true("field 'qty'" in message, message)
    print("  test_error_inside_a_list_names_the_element passed")


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

    print("Combinator types:")
    test_serialize_dict_string_int()
    test_serialize_dict_string_string()
    test_serialize_list_optional_int()
    test_serialize_optional_list_int_present()
    test_serialize_optional_list_int_none()
    test_serialize_list_list_int()
    test_round_trip_combinator_box()
    test_round_trip_combinator_box_null_optional_list()
    print()

    print("Sized integers:")
    test_serialize_sized_ints()
    test_deserialize_sized_ints()
    test_round_trip_sized_ints()
    test_serialize_int32_negative()
    print()

    print("Generic list elements:")
    test_serialize_list_of_structs()
    test_serialize_empty_list_of_structs()
    test_serialize_list_of_structs_nested_deeper()
    test_deserialize_list_of_structs()
    test_error_inside_a_list_names_the_element()
    test_serialize_list_of_float64()
    test_round_trip_list_of_float64()
    print()

    print("All reflection serde tests passed!")
