# Example 10: JSON Schema Validation
#
# Validate JSON documents against schemas to ensure data quality.
# Supports a subset of JSON Schema draft-07.

from src import loads, validate, is_valid


fn main() raises:
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
    
    var user_schema = loads('''
    {
        "type": "object",
        "required": ["name", "email"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "email": {"type": "string"},
            "age": {"type": "integer", "minimum": 0}
        }
    }
    ''')
    
    var valid_user = loads('{"name": "Alice", "email": "alice@example.com", "age": 30}')
    var missing_email = loads('{"name": "Bob"}')
    var invalid_age = loads('{"name": "Charlie", "email": "c@x.com", "age": -5}')
    
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
    if not result.valid:
        print("   Errors:")
        for i in range(len(result.errors)):
            print("     - Path:", result.errors[i].path, "| Message:", result.errors[i].message)
    print()
    
    # ==========================================================
    # 4. Number constraints
    # ==========================================================
    print("4. Number constraints:")
    
    var number_schema = loads('''
    {
        "type": "number",
        "minimum": 0,
        "maximum": 100
    }
    ''')
    
    print("   Schema: 0 <= number <= 100")
    print("   50 valid?", is_valid(loads("50"), number_schema))
    print("   -10 valid?", is_valid(loads("-10"), number_schema))
    print("   150 valid?", is_valid(loads("150"), number_schema))
    print()
    
    # ==========================================================
    # 5. String constraints
    # ==========================================================
    print("5. String constraints:")
    
    var string_schema = loads('''
    {
        "type": "string",
        "minLength": 3,
        "maxLength": 10
    }
    ''')
    
    print("   Schema: 3 <= length <= 10")
    print("   'hello' valid?", is_valid(loads('"hello"'), string_schema))
    print("   'hi' valid?", is_valid(loads('"hi"'), string_schema))
    print("   'verylongstring' valid?", is_valid(loads('"verylongstring"'), string_schema))
    print()
    
    # ==========================================================
    # 6. Array validation
    # ==========================================================
    print("6. Array validation:")
    
    var array_schema = loads('''
    {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 1,
        "maxItems": 5
    }
    ''')
    
    print("   Schema: array of integers, 1-5 items")
    print("   [1,2,3] valid?", is_valid(loads("[1,2,3]"), array_schema))
    print("   [] valid?", is_valid(loads("[]"), array_schema))
    print("   [1,'two'] valid?", is_valid(loads('[1,"two"]'), array_schema))
    print()
    
    # ==========================================================
    # 7. Enum values
    # ==========================================================
    print("7. Enum values:")
    
    var enum_schema = loads('''
    {
        "enum": ["pending", "active", "completed"]
    }
    ''')
    
    print("   Schema: one of [pending, active, completed]")
    print("   'active' valid?", is_valid(loads('"active"'), enum_schema))
    print("   'deleted' valid?", is_valid(loads('"deleted"'), enum_schema))
    print()
    
    # ==========================================================
    # 8. Composition (allOf, anyOf, oneOf)
    # ==========================================================
    print("8. Schema composition:")
    
    var composed_schema = loads('''
    {
        "allOf": [
            {"type": "object"},
            {"required": ["id"]},
            {"properties": {"id": {"type": "integer"}}}
        ]
    }
    ''')
    
    print("   Schema: allOf [object, has id, id is integer]")
    print("   {id: 1} valid?", is_valid(loads('{"id": 1}'), composed_schema))
    print("   {id: 'a'} valid?", is_valid(loads('{"id": "a"}'), composed_schema))
    print("   {name: 'x'} valid?", is_valid(loads('{"name": "x"}'), composed_schema))
    print()
    
    # ==========================================================
    # 9. Practical example - API request validation
    # ==========================================================
    print("9. Practical example - API request:")
    
    var api_schema = loads('''
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
    ''')
    
    var good_request = loads('{"action": "create", "payload": {"name": "test"}}')
    var bad_action = loads('{"action": "invalid", "payload": {}}')
    var extra_field = loads('{"action": "create", "payload": {}, "extra": true}')
    
    print("   Valid request:", is_valid(good_request, api_schema))
    print("   Invalid action:", is_valid(bad_action, api_schema))
    print("   Extra field:", is_valid(extra_field, api_schema))
    print()
    
    print("Done!")
