"""Tests for Serializable/Deserializable traits and serialize/deserialize functions."""

from std.testing import assert_equal, assert_true, assert_raises
from json import (
    loads,
    to_json_value,
    get_string,
    get_int,
    get_bool,
    get_float,
    Value,
)
from json.serialize import Serializable, serialize
from json.deserialize import Deserializable, deserialize


@fieldwise_init
struct Person(Copyable, Deserializable, Serializable):
    """Test struct with both serialization and deserialization."""

    var name: String
    var age: Int
    var active: Bool

    def to_json(self) -> String:
        return (
            '{"name":'
            + to_json_value(self.name)
            + ',"age":'
            + to_json_value(self.age)
            + ',"active":'
            + to_json_value(self.active)
            + "}"
        )

    @staticmethod
    def from_json(json: Value) raises -> Self:
        return Self(
            name=get_string(json, "name"),
            age=get_int(json, "age"),
            active=get_bool(json, "active"),
        )


@fieldwise_init
struct Product(Copyable, Deserializable, Serializable):
    """Test struct with mixed types."""

    var name: String
    var price: Float64
    var quantity: Int

    def to_json(self) -> String:
        return (
            '{"name":'
            + to_json_value(self.name)
            + ',"price":'
            + to_json_value(self.price)
            + ',"quantity":'
            + to_json_value(self.quantity)
            + "}"
        )

    @staticmethod
    def from_json(json: Value) raises -> Self:
        return Self(
            name=get_string(json, "name"),
            price=get_float(json, "price"),
            quantity=get_int(json, "quantity"),
        )


def test_serialize() raises:
    """Test serialize() function."""
    var person = Person(name="Alice", age=30, active=True)
    var json_str = serialize(person)

    # Verify it's valid JSON
    var parsed = loads(json_str)
    assert_true(parsed.is_object())
    assert_equal(get_string(parsed, "name"), "Alice")
    assert_equal(get_int(parsed, "age"), 30)
    assert_equal(get_bool(parsed, "active"), True)
    print("✓ test_serialize passed")


def test_deserialize() raises:
    """Test deserialize() function."""
    var json_str = '{"name":"Bob","age":25,"active":false}'
    var person = deserialize[Person](json_str)

    assert_equal(person.name, "Bob")
    assert_equal(person.age, 25)
    assert_equal(person.active, False)
    print("✓ test_deserialize passed")


def test_round_trip() raises:
    """Test full round-trip: object -> JSON -> object."""
    var original = Person(name="Charlie", age=35, active=True)

    # Serialize
    var json_str = serialize(original)

    # Deserialize
    var restored = deserialize[Person](json_str)

    # Verify
    assert_equal(restored.name, original.name)
    assert_equal(restored.age, original.age)
    assert_equal(restored.active, original.active)
    print("✓ test_round_trip passed")


def test_product_round_trip() raises:
    """Test round-trip with float fields."""
    var original = Product(name="Widget", price=29.99, quantity=100)

    var json_str = serialize(original)
    var restored = deserialize[Product](json_str)

    assert_equal(restored.name, original.name)
    assert_equal(restored.price, original.price)
    assert_equal(restored.quantity, original.quantity)
    print("✓ test_product_round_trip passed")


def test_deserialize_cpu() raises:
    """Test deserialize with CPU backend (default)."""
    var json_str = '{"name":"Dave","age":40,"active":true}'
    var person = deserialize[Person](json_str)

    assert_equal(person.name, "Dave")
    assert_equal(person.age, 40)
    print("✓ test_deserialize_cpu passed")


def test_string_escaping() raises:
    """Test that special characters are properly escaped."""
    var person = Person(name='Alice "Wonder"', age=30, active=True)
    var json_str = serialize(person)

    # Should contain escaped quotes
    var restored = deserialize[Person](json_str)
    assert_equal(restored.name, 'Alice "Wonder"')
    print("✓ test_string_escaping passed")


def test_error_handling() raises:
    """Test error handling for invalid JSON."""
    var json_str = '{"name":"Invalid","age":"not_a_number","active":true}'

    try:
        var person = deserialize[Person](json_str)
        raise Error("Should have failed on invalid data")
    except:
        # Expected to fail
        pass

    print("✓ test_error_handling passed")


def main() raises:
    print("Running Serializable/Deserializable tests...")
    print()

    test_serialize()
    test_deserialize()
    test_round_trip()
    test_product_round_trip()
    test_deserialize_cpu()
    test_string_escaping()
    test_error_handling()

    print()
    print("All serialize/deserialize tests passed!")
