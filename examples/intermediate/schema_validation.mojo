# JSON Schema validation
#
# Validate JSON documents against schemas to ensure data quality.
# Implements JSON Schema draft 2020-12.

from json import loads, validate, is_valid
from json.schema import Schema


def main() raises:
    print("JSON Schema Validation Examples")
    print("=" * 50)
    print()

    # ==========================================================
    # 1. Basic type validation
    # ==========================================================
    print("1. Basic type validation:")

    var type_schema = loads('{"type": "string"}')

    print("   Schema: type=string")
    print("   'hello' valid?", is_valid(loads('"hello"'), type_schema))
    print("   42 valid?", is_valid(loads("42"), type_schema))
    print()

    # ==========================================================
    # 2. Object with required fields
    # ==========================================================
    print("2. Object with required fields:")

    var user_schema = loads(
        """
    {
        "type": "object",
        "required": ["name", "email"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "email": {"type": "string"},
            "age": {"type": "integer", "minimum": 0}
        }
    }
    """
    )

    var valid_user = loads(
        '{"name": "Alice", "email": "alice@example.com", "age": 30}'
    )
    var missing_email = loads('{"name": "Bob"}')
    var invalid_age = loads(
        '{"name": "Charlie", "email": "c@x.com", "age": -5}'
    )

    print("   Complete user valid?", is_valid(valid_user, user_schema))
    print("   Missing email valid?", is_valid(missing_email, user_schema))
    print("   Negative age valid?", is_valid(invalid_age, user_schema))
    print()

    # ==========================================================
    # 3. Detailed error messages
    # ==========================================================
    print("3. Detailed error messages:")

    var result = validate(missing_email, user_schema)
    print("   Validation result: valid=", result.valid)
    if not result:
        print("   Errors:")
        for i in range(len(result.errors)):
            print(
                "     - Instance:",
                result.errors[i].path if result.errors[i].path != "" else "#",
                "| Keyword:",
                result.errors[i].keyword_location,
                "| Message:",
                result.errors[i].message,
            )
    print()

    # ==========================================================
    # 4. Number constraints
    # ==========================================================
    print("4. Number constraints:")

    var number_schema = loads(
        """
    {
        "type": "number",
        "minimum": 0,
        "maximum": 100
    }
    """
    )

    print("   Schema: 0 <= number <= 100")
    print("   50 valid?", is_valid(loads("50"), number_schema))
    print("   -10 valid?", is_valid(loads("-10"), number_schema))
    print("   150 valid?", is_valid(loads("150"), number_schema))
    print()

    # ==========================================================
    # 5. String constraints
    # ==========================================================
    print("5. String constraints:")

    var string_schema = loads(
        """
    {
        "type": "string",
        "minLength": 3,
        "maxLength": 10
    }
    """
    )

    print("   Schema: 3 <= length <= 10")
    print("   'hello' valid?", is_valid(loads('"hello"'), string_schema))
    print("   'hi' valid?", is_valid(loads('"hi"'), string_schema))
    print(
        "   'verylongstring' valid?",
        is_valid(loads('"verylongstring"'), string_schema),
    )
    print()

    # ==========================================================
    # 6. Array validation
    # ==========================================================
    print("6. Array validation:")

    var array_schema = loads(
        """
    {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 1,
        "maxItems": 5
    }
    """
    )

    print("   Schema: array of integers, 1-5 items")
    print("   [1,2,3] valid?", is_valid(loads("[1,2,3]"), array_schema))
    print("   [] valid?", is_valid(loads("[]"), array_schema))
    print("   [1,'two'] valid?", is_valid(loads('[1,"two"]'), array_schema))
    print()

    # ==========================================================
    # 7. Enum values
    # ==========================================================
    print("7. Enum values:")

    var enum_schema = loads(
        """
    {
        "enum": ["pending", "active", "completed"]
    }
    """
    )

    print("   Schema: one of [pending, active, completed]")
    print("   'active' valid?", is_valid(loads('"active"'), enum_schema))
    print("   'deleted' valid?", is_valid(loads('"deleted"'), enum_schema))
    print()

    # ==========================================================
    # 8. Composition (allOf, anyOf, oneOf)
    # ==========================================================
    print("8. Schema composition:")

    var composed_schema = loads(
        """
    {
        "allOf": [
            {"type": "object"},
            {"required": ["id"]},
            {"properties": {"id": {"type": "integer"}}}
        ]
    }
    """
    )

    print("   Schema: allOf [object, has id, id is integer]")
    print("   {id: 1} valid?", is_valid(loads('{"id": 1}'), composed_schema))
    print(
        "   {id: 'a'} valid?", is_valid(loads('{"id": "a"}'), composed_schema)
    )
    print(
        "   {name: 'x'} valid?",
        is_valid(loads('{"name": "x"}'), composed_schema),
    )
    print()

    # ==========================================================
    # 9. Practical example - API request validation
    # ==========================================================
    print("9. Practical example - API request:")

    var api_schema = loads(
        """
    {
        "type": "object",
        "required": ["action", "payload"],
        "properties": {
            "action": {"enum": ["create", "update", "delete"]},
            "payload": {"type": "object"},
            "timestamp": {"type": "string"}
        },
        "additionalProperties": false
    }
    """
    )

    var good_request = loads(
        '{"action": "create", "payload": {"name": "test"}}'
    )
    var bad_action = loads('{"action": "invalid", "payload": {}}')
    var extra_field = loads(
        '{"action": "create", "payload": {}, "extra": true}'
    )

    print("   Valid request:", is_valid(good_request, api_schema))
    print("   Invalid action:", is_valid(bad_action, api_schema))
    print("   Extra field:", is_valid(extra_field, api_schema))
    print()

    # ==========================================================
    # 10. Regular expressions and 2020-12 array keywords
    # ==========================================================
    print("10. Patterns, prefixItems and contains:")

    var pattern_schema = loads(
        '{"type": "string", "pattern": "^[A-Z]{2}-[0-9]{4}$"}'
    )
    print("   Schema: pattern ^[A-Z]{2}-[0-9]{4}$")
    print("   'AB-1234' valid?", is_valid(loads('"AB-1234"'), pattern_schema))
    print("   'ab-1234' valid?", is_valid(loads('"ab-1234"'), pattern_schema))

    var tuple_schema = loads(
        """
    {
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "number"}],
        "items": {"type": "boolean"},
        "contains": {"const": true},
        "minContains": 1
    }
    """
    )

    print("   Schema: [string, number, then booleans, at least one true]")
    print(
        "   ['a', 1, true] valid?",
        is_valid(loads('["a", 1, true]'), tuple_schema),
    )
    print(
        "   ['a', 1, false] valid?",
        is_valid(loads('["a", 1, false]'), tuple_schema),
    )
    print("   ['a', 1, 2] valid?", is_valid(loads('["a", 1, 2]'), tuple_schema))
    print()

    # ==========================================================
    # 11. $ref and $defs, including a recursive schema
    # ==========================================================
    print("11. $ref and $defs:")

    var tree_schema = loads(
        """
    {
        "$id": "https://example.com/tree",
        "$defs": {
            "node": {
                "type": "object",
                "required": ["value"],
                "properties": {
                    "value": {"type": "integer"},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/node"}}
                }
            }
        },
        "$ref": "#/$defs/node"
    }
    """
    )

    var tree = loads(
        '{"value": 1, "children": [{"value": 2}, {"value": 3, "children": []}]}'
    )
    var bad_tree = loads('{"value": 1, "children": [{"value": "two"}]}')

    print("   Well-formed tree valid?", is_valid(tree, tree_schema))
    print("   Tree with a string leaf valid?", is_valid(bad_tree, tree_schema))
    print()

    # ==========================================================
    # 12. if/then/else and unevaluatedProperties
    # ==========================================================
    print("12. Conditional subschemas:")

    var payment_schema = loads(
        """
    {
        "type": "object",
        "properties": {"method": {"enum": ["card", "cash"]}},
        "if": {"properties": {"method": {"const": "card"}}, "required": ["method"]},
        "then": {"properties": {"pan": {"type": "string"}}, "required": ["pan"]},
        "else": {"properties": {"received": {"type": "number"}}},
        "unevaluatedProperties": false
    }
    """
    )

    print(
        "   card without a pan valid?",
        is_valid(loads('{"method": "card"}'), payment_schema),
    )
    print(
        "   card with a pan valid?",
        is_valid(loads('{"method": "card", "pan": "4111"}'), payment_schema),
    )
    print(
        "   cash with received valid?",
        is_valid(loads('{"method": "cash", "received": 20}'), payment_schema),
    )
    print(
        "   cash with a stray key valid?",
        is_valid(loads('{"method": "cash", "tip": 2}'), payment_schema),
    )
    print()

    # ==========================================================
    # 13. format, an annotation by default
    # ==========================================================
    print("13. format:")

    var contact_schema = loads(
        '{"type": "object", "properties": {"email": {"format": "email"}}}'
    )
    var contact = loads('{"email": "not an address"}')

    var annotated = validate(contact, contact_schema)
    print("   Collected as an annotation, so valid?", annotated.valid)
    for i in range(len(annotated.format_annotations)):
        print(
            "     -",
            annotated.format_annotations[i].path,
            "is a valid",
            annotated.format_annotations[i].format + "?",
            annotated.format_annotations[i].matched,
        )

    var asserting = Schema.compile(contact_schema, assert_format=True)
    print("   Asserted, so valid?", asserting.is_valid(contact))
    print()

    # ==========================================================
    # 14. Compile once, validate many
    # ==========================================================
    print("14. Compile once, validate many:")

    var compiled = Schema.compile(user_schema)
    var batch = loads(
        """
    [
        {"name": "Alice", "email": "alice@example.com", "age": 30},
        {"name": "Bob"},
        {"name": "Carol", "email": "carol@example.com", "age": -5}
    ]
    """
    )

    var records = batch.array_items()
    for i in range(len(records)):
        print("   record", i, "valid?", compiled.is_valid(records[i]))
    print()

    print("Done!")
