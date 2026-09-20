"""Example: Struct serialization and deserialization with traits.

This example shows how to implement Serializable and Deserializable traits
for your structs to enable clean serialize/deserialize functions.
"""

from json import loads, Value
from json.serialize import Serializable, serialize, to_json_value
from json.deserialize import Deserializable, deserialize, get_string, get_int, get_bool


@fieldwise_init
struct Person(Serializable, Deserializable, Copyable):
    """A person with name, age, and active status.

    Implements both Serializable and Deserializable for full round-trip support.
    """

    var name: String
    var age: Int
    var active: Bool

    def to_json(self) -> String:
        """Serialize to JSON string."""
        return (
            '{"name":'
            + to_json_value(self.name)
            + ',"age":'
            + to_json_value(self.age)
            + ',"active":'
            + to_json_value(self.active)
            + '}'
        )

    @staticmethod
    def from_json(json: Value) raises -> Self:
        """Deserialize from JSON Value."""
        return Self(
            name=get_string(json, "name"),
            age=get_int(json, "age"),
            active=get_bool(json, "active"),
        )


def example_serialize():
    """Demonstrate serialization."""
    print("=== Serialization ===\n")

    var person = Person(name="Alice", age=30, active=True)
    print("Original object:", person.name, person.age, person.active)

    # Serialize using the serialize() helper
    var json_str = serialize(person)
    print("Serialized JSON:", json_str)
    print()


def example_deserialize() raises:
    """Demonstrate deserialization."""
    print("=== Deserialization ===\n")

    var json_str = '{"name":"Bob","age":25,"active":false}'
    print("JSON string:", json_str)

    # Deserialize using the deserialize() helper
    var person = deserialize[Person](json_str)
    print("Deserialized object:", person.name, person.age, person.active)
    print()


def example_round_trip() raises:
    """Demonstrate full round-trip."""
    print("=== Round-Trip ===\n")

    var original = Person(name="Charlie", age=35, active=True)
    print("Original:", original.name, original.age, original.active)

    # Serialize
    var json_str = serialize(original)
    print("JSON:", json_str)

    # Deserialize
    var restored = deserialize[Person](json_str)
    print("Restored:", restored.name, restored.age, restored.active)

    # Verify
    var matches = (
        restored.name == original.name
        and restored.age == original.age
        and restored.active == original.active
    )
    print("Round-trip successful:", matches)
    print()


def example_direct_method_calls() raises:
    """Show that you can also call methods directly."""
    print("=== Direct Method Calls ===\n")

    # You don't have to use serialize/deserialize helpers
    # You can call to_json/from_json directly

    var person = Person(name="Eve", age=28, active=False)
    print("Original:", person.name)

    # Direct serialization
    var json_str = person.to_json()
    print("JSON:", json_str)

    # Direct deserialization
    var json = loads(json_str)
    var restored = Person.from_json(json)
    print("Restored:", restored.name)
    print()


def main() raises:
    print("\n╔════════════════════════════════════════════╗")
    print("║  Struct Serialization/Deserialization     ║")
    print("╚════════════════════════════════════════════╝\n")

    example_serialize()
    example_deserialize()
    example_round_trip()
    example_direct_method_calls()

    print("═" * 46)
    print("✓ All examples completed successfully!")
    print("═" * 46)
