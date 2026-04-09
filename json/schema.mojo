# json - JSON Schema validation (draft-07 subset)
#
# Validates JSON documents against JSON Schema.
# Implements a useful subset of JSON Schema draft-07.

from std.collections import List
from .value import Value, Null
from .parser import loads
from .serialize import dumps


struct ValidationError(Copyable, Movable):
    """A validation error with path and message."""

    var path: String
    var message: String

    def __init__(out self, path: String, message: String):
        self.path = path
        self.message = message

    def __str__(self) -> String:
        if self.path == "":
            return self.message
        return self.path + ": " + self.message


struct ValidationResult(Movable):
    """Result of schema validation."""

    var valid: Bool
    var errors: List[ValidationError]

    def __init__(out self):
        self.valid = True
        self.errors = List[ValidationError]()

    def add_error(mut self, path: String, message: String):
        self.valid = False
        self.errors.append(ValidationError(path, message))


def validate(document: Value, schema: Value) raises -> ValidationResult:
    """Validate a JSON document against a JSON Schema.

    Supported schema keywords:
    - type: "null", "boolean", "integer", "number", "string", "array", "object"
    - enum: Array of allowed values
    - const: Exact value match
    - minimum, maximum: Number bounds
    - minLength, maxLength: String length
    - pattern: String regex (basic support)
    - minItems, maxItems: Array length
    - items: Schema for array items
    - minProperties, maxProperties: Object size
    - properties: Schema for object properties
    - required: Required property names
    - additionalProperties: Allow extra properties
    - anyOf, oneOf, allOf: Schema composition
    - not: Negation

    Args:
        document: The JSON document to validate.
        schema: The JSON Schema.

    Returns:
        ValidationResult with valid flag and error list.

    Example:
        var schema = loads('{"type":"object","required":["name"]}')
        var doc = loads('{"name":"Alice"}')
        var result = validate(doc, schema)
        if result.valid:
            print("Valid!").
    """
    var result = ValidationResult()
    _validate_value(document, schema, "", result)
    return result^


def is_valid(document: Value, schema: Value) raises -> Bool:
    """Check if a document is valid against a schema.

    Args:
        document: The JSON document to validate.
        schema: The JSON Schema.

    Returns:
        True if valid, False otherwise.
    """
    var result = validate(document, schema)
    return result.valid


def _validate_value(
    value: Value,
    schema: Value,
    path: String,
    mut result: ValidationResult,
):
    """Validate a value against a schema."""
    # Boolean schema
    if schema.is_bool():
        if not schema.bool_value():
            result.add_error(path, "Schema is false, nothing is valid")
        return

    if not schema.is_object():
        return  # Empty or non-object schema validates everything

    # type validation
    try:
        var type_val = schema["type"]
        if type_val.is_string():
            _validate_type(value, type_val.string_value(), path, result)
        elif type_val.is_array():
            _validate_type_array(value, type_val, path, result)
    except:
        pass  # No type constraint

    # enum validation
    try:
        var enum_val = schema["enum"]
        _validate_enum(value, enum_val, path, result)
    except:
        pass

    # const validation
    try:
        var const_val = schema["const"]
        _validate_const(value, const_val, path, result)
    except:
        pass

    # Number validations
    if value.is_int() or value.is_float():
        _validate_number(value, schema, path, result)

    # String validations
    if value.is_string():
        _validate_string(value, schema, path, result)

    # Array validations
    if value.is_array():
        _validate_array(value, schema, path, result)

    # Object validations
    if value.is_object():
        _validate_object(value, schema, path, result)

    # Composition keywords
    try:
        var all_of = schema["allOf"]
        _validate_all_of(value, all_of, path, result)
    except:
        pass

    try:
        var any_of = schema["anyOf"]
        _validate_any_of(value, any_of, path, result)
    except:
        pass

    try:
        var one_of = schema["oneOf"]
        _validate_one_of(value, one_of, path, result)
    except:
        pass

    try:
        var not_schema = schema["not"]
        _validate_not(value, not_schema, path, result)
    except:
        pass


def _validate_type(
    value: Value,
    type_name: String,
    path: String,
    mut result: ValidationResult,
):
    """Validate value matches expected type."""
    var valid = False

    if type_name == "null":
        valid = value.is_null()
    elif type_name == "boolean":
        valid = value.is_bool()
    elif type_name == "integer":
        valid = value.is_int()
    elif type_name == "number":
        valid = value.is_int() or value.is_float()
    elif type_name == "string":
        valid = value.is_string()
    elif type_name == "array":
        valid = value.is_array()
    elif type_name == "object":
        valid = value.is_object()

    if not valid:
        result.add_error(path, "Expected type " + type_name)


def _validate_type_array(
    value: Value,
    types: Value,
    path: String,
    mut result: ValidationResult,
):
    """Validate value matches one of multiple types."""
    try:
        var type_list = types.array_items()
        for i in range(len(type_list)):
            var type_val = type_list[i].copy()
            if type_val.is_string():
                var temp_result = ValidationResult()
                _validate_type(
                    value, type_val.string_value(), path, temp_result
                )
                if temp_result.valid:
                    return  # Found a matching type

        result.add_error(path, "Value does not match any of the allowed types")
    except:
        pass


def _validate_enum(
    value: Value,
    enum_val: Value,
    path: String,
    mut result: ValidationResult,
):
    """Validate value is in enum list."""
    try:
        var items = enum_val.array_items()
        for i in range(len(items)):
            if _values_equal(value, items[i]):
                return
        result.add_error(path, "Value not in enum")
    except:
        pass


def _validate_const(
    value: Value,
    const_val: Value,
    path: String,
    mut result: ValidationResult,
):
    """Validate value equals const."""
    if not _values_equal(value, const_val):
        result.add_error(path, "Value does not equal const")


def _validate_number(
    value: Value,
    schema: Value,
    path: String,
    mut result: ValidationResult,
):
    """Validate number constraints."""
    var num = value.float_value() if value.is_float() else Float64(
        value.int_value()
    )

    try:
        var min_val = schema["minimum"]
        var min_num = min_val.float_value() if min_val.is_float() else Float64(
            min_val.int_value()
        )
        if num < min_num:
            result.add_error(path, "Value below minimum " + String(min_num))
    except:
        pass

    try:
        var max_val = schema["maximum"]
        var max_num = max_val.float_value() if max_val.is_float() else Float64(
            max_val.int_value()
        )
        if num > max_num:
            result.add_error(path, "Value above maximum " + String(max_num))
    except:
        pass

    try:
        var exc_min = schema["exclusiveMinimum"]
        var min_num = exc_min.float_value() if exc_min.is_float() else Float64(
            exc_min.int_value()
        )
        if num <= min_num:
            result.add_error(
                path, "Value must be greater than " + String(min_num)
            )
    except:
        pass

    try:
        var exc_max = schema["exclusiveMaximum"]
        var max_num = exc_max.float_value() if exc_max.is_float() else Float64(
            exc_max.int_value()
        )
        if num >= max_num:
            result.add_error(path, "Value must be less than " + String(max_num))
    except:
        pass

    try:
        var multiple = schema["multipleOf"]
        var mult_num = (
            multiple.float_value() if multiple.is_float() else Float64(
                multiple.int_value()
            )
        )
        if mult_num != 0:
            var remainder = num - (Float64(Int(num / mult_num)) * mult_num)
            if remainder != 0:
                result.add_error(
                    path, "Value must be multiple of " + String(mult_num)
                )
    except:
        pass


def _validate_string(
    value: Value,
    schema: Value,
    path: String,
    mut result: ValidationResult,
):
    """Validate string constraints."""
    var s = value.string_value()
    var length = s.byte_length()

    try:
        var min_len = schema["minLength"]
        if length < Int(min_len.int_value()):
            result.add_error(
                path, "String too short, minimum " + String(min_len.int_value())
            )
    except:
        pass

    try:
        var max_len = schema["maxLength"]
        if length > Int(max_len.int_value()):
            result.add_error(
                path, "String too long, maximum " + String(max_len.int_value())
            )
    except:
        pass

    # Basic pattern matching (exact match only for now)
    try:
        var pattern = schema["pattern"]
        var pat = pattern.string_value()
        # Simple pattern matching - just check if pattern is contained
        # Full regex support would require a regex library
        if pat.startswith("^") and pat.endswith("$"):
            # Exact match
            var inner = String(
                String(unsafe_from_utf8=pat.as_bytes()[1 : pat.byte_length() - 1])
            )
            if s != inner:
                result.add_error(path, "String does not match pattern")
        elif not s.find(pat) >= 0:
            # Contains check
            pass  # Skip complex patterns
    except:
        pass


def _validate_array(
    value: Value,
    schema: Value,
    path: String,
    mut result: ValidationResult,
):
    """Validate array constraints."""
    try:
        var items = value.array_items()
        var count = len(items)

        try:
            var min_items = schema["minItems"]
            if count < Int(min_items.int_value()):
                result.add_error(
                    path,
                    "Array too short, minimum "
                    + String(min_items.int_value())
                    + " items",
                )
        except:
            pass

        try:
            var max_items = schema["maxItems"]
            if count > Int(max_items.int_value()):
                result.add_error(
                    path,
                    "Array too long, maximum "
                    + String(max_items.int_value())
                    + " items",
                )
        except:
            pass

        try:
            var unique = schema["uniqueItems"]
            if unique.is_bool() and unique.bool_value():
                for i in range(count):
                    for j in range(i + 1, count):
                        if _values_equal(items[i], items[j]):
                            result.add_error(
                                path, "Array contains duplicate items"
                            )
                            break
        except:
            pass

        # items schema
        try:
            var items_schema = schema["items"]
            if items_schema.is_object() or items_schema.is_bool():
                for i in range(count):
                    var item_path = path + "/" + String(i)
                    _validate_value(items[i], items_schema, item_path, result)
            elif items_schema.is_array():
                var schemas = items_schema.array_items()
                for i in range(min(count, len(schemas))):
                    var item_path = path + "/" + String(i)
                    _validate_value(items[i], schemas[i], item_path, result)
        except:
            pass
    except:
        pass


def _validate_object(
    value: Value,
    schema: Value,
    path: String,
    mut result: ValidationResult,
):
    """Validate object constraints."""
    var items: List[Tuple[String, Value]]
    try:
        items = value.object_items()
    except:
        return

    var count = len(items)
    var keys = List[String]()
    for i in range(count):
        keys.append(items[i][0])

    try:
        var min_props = schema["minProperties"]
        if count < Int(min_props.int_value()):
            result.add_error(
                path,
                "Object has too few properties, minimum "
                + String(min_props.int_value()),
            )
    except:
        pass

    try:
        var max_props = schema["maxProperties"]
        if count > Int(max_props.int_value()):
            result.add_error(
                path,
                "Object has too many properties, maximum "
                + String(max_props.int_value()),
            )
    except:
        pass

    # required
    try:
        var required = schema["required"]
        var req_list = required.array_items()
        for i in range(len(req_list)):
            var req_key = req_list[i].string_value()
            var found = False
            for j in range(len(keys)):
                if keys[j] == req_key:
                    found = True
                    break
            if not found:
                result.add_error(path, "Missing required property: " + req_key)
    except:
        pass

    # properties
    var validated_keys = List[String]()
    try:
        var properties = schema["properties"]
        var prop_items = properties.object_items()
        for i in range(len(prop_items)):
            var prop_key = prop_items[i][0]
            var prop_schema = prop_items[i][1].copy()
            validated_keys.append(prop_key)

            # Find the property in the value
            for j in range(len(items)):
                if items[j][0] == prop_key:
                    var prop_path = path + "/" + prop_key
                    _validate_value(items[j][1], prop_schema, prop_path, result)
                    break
    except:
        pass

    # additionalProperties
    try:
        var additional = schema["additionalProperties"]
        if additional.is_bool() and not additional.bool_value():
            # No additional properties allowed
            for i in range(len(keys)):
                var key = keys[i]
                var is_validated = False
                for j in range(len(validated_keys)):
                    if validated_keys[j] == key:
                        is_validated = True
                        break
                if not is_validated:
                    result.add_error(
                        path, "Additional property not allowed: " + key
                    )
        elif additional.is_object():
            # Validate additional properties against schema
            for i in range(len(items)):
                var key = items[i][0]
                var is_validated = False
                for j in range(len(validated_keys)):
                    if validated_keys[j] == key:
                        is_validated = True
                        break
                if not is_validated:
                    var prop_path = path + "/" + key
                    _validate_value(items[i][1], additional, prop_path, result)
    except:
        pass


def _validate_all_of(
    value: Value,
    schemas: Value,
    path: String,
    mut result: ValidationResult,
):
    """Validate value matches all schemas."""
    try:
        var schema_list = schemas.array_items()
        for i in range(len(schema_list)):
            _validate_value(value, schema_list[i], path, result)
    except:
        pass


def _validate_any_of(
    value: Value,
    schemas: Value,
    path: String,
    mut result: ValidationResult,
):
    """Validate value matches at least one schema."""
    try:
        var schema_list = schemas.array_items()
        for i in range(len(schema_list)):
            var temp_result = ValidationResult()
            _validate_value(value, schema_list[i], path, temp_result)
            if temp_result.valid:
                return  # Found a match
        result.add_error(path, "Value does not match any schema in anyOf")
    except:
        pass


def _validate_one_of(
    value: Value,
    schemas: Value,
    path: String,
    mut result: ValidationResult,
):
    """Validate value matches exactly one schema."""
    try:
        var schema_list = schemas.array_items()
        var match_count = 0

        for i in range(len(schema_list)):
            var temp_result = ValidationResult()
            _validate_value(value, schema_list[i], path, temp_result)
            if temp_result.valid:
                match_count += 1

        if match_count == 0:
            result.add_error(path, "Value does not match any schema in oneOf")
        elif match_count > 1:
            result.add_error(
                path, "Value matches more than one schema in oneOf"
            )
    except:
        pass


def _validate_not(
    value: Value,
    schema: Value,
    path: String,
    mut result: ValidationResult,
):
    """Validate value does NOT match schema."""
    var temp_result = ValidationResult()
    _validate_value(value, schema, path, temp_result)
    if temp_result.valid:
        result.add_error(path, "Value should not match schema in 'not'")


def _values_equal(a: Value, b: Value) -> Bool:
    """Check if two values are equal."""
    if a.is_null() and b.is_null():
        return True
    if a.is_bool() and b.is_bool():
        return a.bool_value() == b.bool_value()
    if a.is_int() and b.is_int():
        return a.int_value() == b.int_value()
    if a.is_float() and b.is_float():
        return a.float_value() == b.float_value()
    if a.is_string() and b.is_string():
        return a.string_value() == b.string_value()
    if a.is_array() and b.is_array():
        return dumps(a) == dumps(b)
    if a.is_object() and b.is_object():
        return dumps(a) == dumps(b)
    # Cross-type number comparison
    if (a.is_int() or a.is_float()) and (b.is_int() or b.is_float()):
        var af = a.float_value() if a.is_float() else Float64(a.int_value())
        var bf = b.float_value() if b.is_float() else Float64(b.int_value())
        return af == bf
    return False
