# Tests for json/schema.mojo -- JSON Schema draft 2020-12.
#
# Three things need pinning here.
#
# First, every keyword in both directions. A validator that only ever
# sees instances it accepts is indistinguishable from one that accepts
# everything, which is the state this module was in, so each keyword
# gets a passing instance and a failing one.
#
# Second, the ten behaviours the previous version got wrong. Those have
# a section of their own below, because each of them returned "valid"
# for an instance a conforming validator rejects, and a regression in
# any of them is silent by construction.
#
# Third, the schema errors. A malformed schema used to read as "no
# constraint", so the cases that assert a raise are as load-bearing as
# the ones that assert a verdict.

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from json import is_valid, loads, validate
from json.schema import Schema


def _valid(document: String, schema: String) raises -> Bool:
    return is_valid(loads(document), loads(schema))


def _assert_schema_error(schema: String) raises:
    """Assert that the schema is rejected when it is compiled."""
    var raised = False
    try:
        _ = Schema.compile(loads(schema))
    except:
        raised = True
    assert_true(raised, "expected a schema error from " + schema)


# ---------------------------------------------------------------------------
# type
# ---------------------------------------------------------------------------


def test_type_string() raises:
    assert_true(_valid('"hello"', '{"type":"string"}'))
    assert_false(_valid("123", '{"type":"string"}'))


def test_type_integer() raises:
    assert_true(_valid("42", '{"type":"integer"}'))
    assert_false(_valid("3.14", '{"type":"integer"}'))


def test_type_number() raises:
    assert_true(_valid("42", '{"type":"number"}'))
    assert_true(_valid("3.14", '{"type":"number"}'))
    assert_false(_valid('"hello"', '{"type":"number"}'))


def test_type_boolean() raises:
    assert_true(_valid("true", '{"type":"boolean"}'))
    assert_true(_valid("false", '{"type":"boolean"}'))
    assert_false(_valid("1", '{"type":"boolean"}'))


def test_type_null() raises:
    assert_true(_valid("null", '{"type":"null"}'))
    assert_false(_valid("0", '{"type":"null"}'))


def test_type_array() raises:
    assert_true(_valid("[1,2,3]", '{"type":"array"}'))
    assert_false(_valid("{}", '{"type":"array"}'))


def test_type_object() raises:
    assert_true(_valid('{"a":1}', '{"type":"object"}'))
    assert_false(_valid("[]", '{"type":"object"}'))


def test_type_union() raises:
    assert_true(_valid('"a"', '{"type":["string","integer"]}'))
    assert_true(_valid("1", '{"type":["string","integer"]}'))
    assert_false(_valid("true", '{"type":["string","integer"]}'))


def test_boolean_schema() raises:
    assert_true(_valid("1", "true"))
    assert_false(_valid("1", "false"))
    assert_true(_valid("1", "{}"))


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def test_minimum_and_maximum() raises:
    assert_true(_valid("10", '{"minimum":5}'))
    assert_true(_valid("5", '{"minimum":5}'))
    assert_false(_valid("3", '{"minimum":5}'))
    assert_true(_valid("10", '{"maximum":10}'))
    assert_false(_valid("15", '{"maximum":10}'))


def test_exclusive_bounds() raises:
    assert_false(_valid("5", '{"exclusiveMinimum":5}'))
    assert_true(_valid("6", '{"exclusiveMinimum":5}'))
    assert_false(_valid("5", '{"exclusiveMaximum":5}'))
    assert_true(_valid("4", '{"exclusiveMaximum":5}'))


def test_bounds_ignore_non_numbers() raises:
    """A numeric keyword has nothing to say about a string."""
    assert_true(_valid('"abc"', '{"minimum":5,"maximum":1}'))


def test_minimum_is_exact_above_two_to_the_53() raises:
    """Widening both sides to binary64 would make these compare equal."""
    assert_false(_valid("9007199254740992", '{"minimum":9007199254740993}'))
    assert_true(_valid("9007199254740993", '{"minimum":9007199254740993}'))


def test_multiple_of() raises:
    assert_true(_valid("10", '{"multipleOf":5}'))
    assert_false(_valid("11", '{"multipleOf":5}'))
    assert_true(_valid("4.5", '{"multipleOf":1.5}'))
    assert_false(_valid("4.6", '{"multipleOf":1.5}'))
    assert_true(_valid("-10", '{"multipleOf":5}'))


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------


def test_min_and_max_length() raises:
    assert_true(_valid('"hello"', '{"minLength":3}'))
    assert_false(_valid('"hi"', '{"minLength":3}'))
    assert_true(_valid('"hi"', '{"maxLength":5}'))
    assert_false(_valid('"hello world"', '{"maxLength":5}'))


def test_pattern_anchored() raises:
    assert_true(_valid('"12345"', '{"pattern":"^[0-9]+$"}'))
    assert_false(_valid('"12a45"', '{"pattern":"^[0-9]+$"}'))


def test_pattern_is_unanchored() raises:
    """Section 6.3.3 makes a partial match enough."""
    assert_true(_valid('"xx42yy"', '{"pattern":"[0-9]+"}'))
    assert_false(_valid('"xxyy"', '{"pattern":"[0-9]+"}'))


def test_pattern_half_anchored() raises:
    assert_true(_valid('"abcdef"', '{"pattern":"^abc"}'))
    assert_false(_valid('"zabcdef"', '{"pattern":"^abc"}'))
    assert_true(_valid('"zzzdef"', '{"pattern":"def$"}'))
    assert_false(_valid('"defzzz"', '{"pattern":"def$"}'))


def test_pattern_quantifiers_and_classes() raises:
    assert_true(_valid('"AB-1234"', '{"pattern":"^[A-Z]{2}-[0-9]{4}$"}'))
    assert_false(_valid('"AB-123"', '{"pattern":"^[A-Z]{2}-[0-9]{4}$"}'))
    assert_true(_valid('"a1"', '{"pattern":"^\\\\w+$"}'))
    assert_false(_valid('"a b"', '{"pattern":"^\\\\w+$"}'))


def test_pattern_ignores_non_strings() raises:
    assert_true(_valid("42", '{"pattern":"^[a-z]+$"}'))


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------


def test_min_and_max_items() raises:
    assert_true(_valid("[1,2,3]", '{"minItems":2}'))
    assert_false(_valid("[1]", '{"minItems":2}'))
    assert_true(_valid("[1,2]", '{"maxItems":3}'))
    assert_false(_valid("[1,2,3,4]", '{"maxItems":3}'))


def test_items() raises:
    assert_true(_valid("[1,2,3]", '{"items":{"type":"integer"}}'))
    assert_false(_valid('[1,"two",3]', '{"items":{"type":"integer"}}'))


def test_prefix_items() raises:
    var schema = '{"prefixItems":[{"type":"integer"},{"type":"string"}]}'
    assert_true(_valid('[1,"a"]', schema))
    assert_false(_valid('["a",1]', schema))
    # A short array is fine: prefixItems constrains the items that are
    # there, and minItems is the keyword that requires more.
    assert_true(_valid("[1]", schema))
    # Anything past the prefix is unconstrained without `items`.
    assert_true(_valid('[1,"a",true]', schema))


def test_items_applies_past_the_prefix() raises:
    """The 2020-12 change: `items` starts where `prefixItems` stopped."""
    var schema = (
        '{"prefixItems":[{"type":"integer"}],"items":{"type":"string"}}'
    )
    assert_true(_valid('[1,"a","b"]', schema))
    assert_false(_valid('[1,"a",2]', schema))
    # The integer in position 0 would fail `items` if `items` started
    # at the front, which is how draft-07 would have read this.
    assert_true(_valid("[1]", schema))


def test_unique_items() raises:
    assert_true(_valid("[1,2,3]", '{"uniqueItems":true}'))
    assert_false(_valid("[1,2,1]", '{"uniqueItems":true}'))
    assert_true(_valid("[1,2,1]", '{"uniqueItems":false}'))


def test_contains() raises:
    assert_true(_valid('[1,"a"]', '{"contains":{"type":"string"}}'))
    assert_false(_valid("[1,2]", '{"contains":{"type":"string"}}'))


def test_min_and_max_contains() raises:
    var schema = '{"contains":{"type":"string"},"minContains":2}'
    assert_false(_valid('["a"]', schema))
    assert_true(_valid('["a","b"]', schema))
    var capped = '{"contains":{"type":"string"},"maxContains":1}'
    assert_true(_valid('["a",1]', capped))
    assert_false(_valid('["a","b"]', capped))


def test_min_contains_zero() raises:
    """`minContains: 0` makes `contains` unable to fail on its own."""
    assert_true(
        _valid("[1,2]", '{"contains":{"type":"string"},"minContains":0}')
    )


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------


def test_required() raises:
    assert_true(_valid('{"name":"Alice"}', '{"required":["name"]}'))
    assert_false(_valid('{"age":30}', '{"required":["name"]}'))


def test_properties() raises:
    var schema = '{"properties":{"age":{"type":"integer"}}}'
    assert_true(_valid('{"age":30}', schema))
    assert_false(_valid('{"age":"thirty"}', schema))
    # An absent property is not a failure; `required` says otherwise.
    assert_true(_valid("{}", schema))


def test_min_and_max_properties() raises:
    assert_false(_valid('{"a":1}', '{"minProperties":2}'))
    assert_true(_valid('{"a":1,"b":2}', '{"minProperties":2}'))
    assert_false(_valid('{"a":1,"b":2}', '{"maxProperties":1}'))


def test_pattern_properties() raises:
    var schema = '{"patternProperties":{"^x_":{"type":"integer"}}}'
    assert_true(_valid('{"x_1":1}', schema))
    assert_false(_valid('{"x_1":"one"}', schema))
    assert_true(_valid('{"y":"one"}', schema))


def test_additional_properties_false() raises:
    var schema = (
        '{"properties":{"a":{"type":"integer"}},"additionalProperties":false}'
    )
    assert_true(_valid('{"a":1}', schema))
    assert_false(_valid('{"a":1,"b":2}', schema))


def test_additional_properties_schema() raises:
    var schema = (
        '{"properties":{"a":{}},"additionalProperties":{"type":"string"}}'
    )
    assert_true(_valid('{"a":1,"b":"x"}', schema))
    assert_false(_valid('{"a":1,"b":2}', schema))


def test_property_names() raises:
    assert_true(_valid('{"ab":1}', '{"propertyNames":{"maxLength":2}}'))
    assert_false(_valid('{"abc":1}', '{"propertyNames":{"maxLength":2}}'))


def test_dependent_required() raises:
    var schema = '{"dependentRequired":{"card":["pan"]}}'
    assert_true(_valid("{}", schema))
    assert_true(_valid('{"card":1,"pan":"x"}', schema))
    assert_false(_valid('{"card":1}', schema))


def test_dependent_schemas() raises:
    var schema = '{"dependentSchemas":{"card":{"required":["pan"]}}}'
    assert_true(_valid("{}", schema))
    assert_false(_valid('{"card":1}', schema))
    assert_true(_valid('{"card":1,"pan":"x"}', schema))


# ---------------------------------------------------------------------------
# enum and const
# ---------------------------------------------------------------------------


def test_enum() raises:
    var schema = '{"enum":["red","green","blue"]}'
    assert_true(_valid('"red"', schema))
    assert_false(_valid('"yellow"', schema))


def test_const() raises:
    assert_true(_valid("42", '{"const":42}'))
    assert_false(_valid("43", '{"const":42}'))
    assert_true(_valid("null", '{"const":null}'))
    assert_false(_valid("0", '{"const":null}'))


def test_const_distinguishes_boolean_from_number() raises:
    """`true` is not `1`, in either direction."""
    assert_false(_valid("1", '{"const":true}'))
    assert_false(_valid("true", '{"const":1}'))


# ---------------------------------------------------------------------------
# Applicators
# ---------------------------------------------------------------------------


def test_all_of() raises:
    var schema = '{"allOf":[{"type":"object"},{"required":["name"]}]}'
    assert_true(_valid('{"name":"Alice"}', schema))
    assert_false(_valid('{"age":30}', schema))


def test_any_of() raises:
    var schema = '{"anyOf":[{"type":"string"},{"type":"integer"}]}'
    assert_true(_valid('"hello"', schema))
    assert_true(_valid("42", schema))
    assert_false(_valid("true", schema))


def test_one_of() raises:
    var schema = '{"oneOf":[{"type":"integer"},{"type":"number","minimum":10}]}'
    assert_true(_valid("3", schema))
    assert_true(_valid("10.5", schema))
    # 12 is an integer and is at least 10, so it matches both.
    assert_false(_valid("12", schema))
    assert_false(_valid('"x"', schema))


def test_not() raises:
    assert_true(_valid("42", '{"not":{"type":"string"}}'))
    assert_false(_valid('"hello"', '{"not":{"type":"string"}}'))
    assert_true(_valid("42", '{"not":{"not":{"type":"integer"}}}'))


def test_if_then() raises:
    var schema = (
        '{"if":{"properties":{"k":{"const":"a"}},"required":["k"]},'
        '"then":{"required":["z"]}}'
    )
    assert_false(_valid('{"k":"a"}', schema))
    assert_true(_valid('{"k":"a","z":1}', schema))
    assert_true(_valid('{"k":"b"}', schema))


def test_if_else() raises:
    var schema = (
        '{"if":{"properties":{"k":{"const":"a"}},"required":["k"]},'
        '"else":{"required":["z"]}}'
    )
    assert_true(_valid('{"k":"a"}', schema))
    assert_false(_valid('{"k":"b"}', schema))
    assert_true(_valid('{"k":"b","z":1}', schema))


def test_if_without_then_or_else_never_fails() raises:
    assert_true(_valid('"x"', '{"if":{"type":"integer"}}'))


# ---------------------------------------------------------------------------
# Core: $ref, $defs, $anchor, $id
# ---------------------------------------------------------------------------


def test_ref_to_defs() raises:
    var schema = (
        '{"$defs":{"Name":{"type":"string"}},'
        '"properties":{"n":{"$ref":"#/$defs/Name"}}}'
    )
    assert_true(_valid('{"n":"Alice"}', schema))
    assert_false(_valid('{"n":7}', schema))


def test_ref_to_root() raises:
    var schema = '{"type":"object","properties":{"child":{"$ref":"#"}}}'
    assert_true(_valid('{"child":{"child":{}}}', schema))
    assert_false(_valid('{"child":[]}', schema))


def test_ref_to_anchor() raises:
    var schema = (
        '{"$defs":{"N":{"$anchor":"name","type":"string"}},"$ref":"#name"}'
    )
    assert_true(_valid('"Alice"', schema))
    assert_false(_valid("7", schema))


def test_ref_to_id() raises:
    var schema = (
        '{"$id":"https://example.com/root",'
        '"$defs":{"N":{"$id":"https://example.com/name","type":"string"}},'
        '"$ref":"https://example.com/name"}'
    )
    assert_true(_valid('"Alice"', schema))
    assert_false(_valid("7", schema))


def test_ref_pointer_escaping() raises:
    """`~1` in a fragment names a member whose name contains a solidus."""
    var schema = '{"$defs":{"a/b":{"type":"string"}},"$ref":"#/$defs/a~1b"}'
    assert_true(_valid('"x"', schema))
    assert_false(_valid("1", schema))


def test_ref_applies_alongside_siblings() raises:
    """2020-12 keeps the sibling keywords of a `$ref`."""
    var schema = (
        '{"$defs":{"S":{"type":"string"}},"$ref":"#/$defs/S","maxLength":3}'
    )
    assert_true(_valid('"abc"', schema))
    assert_false(_valid('"abcd"', schema))


def test_ref_into_properties() raises:
    var schema = (
        '{"properties":{"a":{"type":"integer"}},'
        '"additionalProperties":{"$ref":"#/properties/a"}}'
    )
    assert_true(_valid('{"a":1,"b":2}', schema))
    assert_false(_valid('{"a":1,"b":"x"}', schema))


def test_ref_cycle_raises() raises:
    """A self-reference with no base case fails loudly, not by stack."""
    var raised = False
    try:
        _ = is_valid(loads("1"), loads('{"$ref":"#"}'))
    except:
        raised = True
    assert_true(raised)


def test_comment_and_schema_keywords_are_inert() raises:
    var schema = (
        '{"$schema":"https://json-schema.org/draft/2020-12/schema",'
        '"$comment":"ignored","type":"integer"}'
    )
    assert_true(_valid("1", schema))
    assert_false(_valid('"x"', schema))


# ---------------------------------------------------------------------------
# Unevaluated locations
# ---------------------------------------------------------------------------


def test_unevaluated_properties() raises:
    var schema = '{"properties":{"a":{}},"unevaluatedProperties":false}'
    assert_true(_valid('{"a":1}', schema))
    assert_false(_valid('{"a":1,"b":2}', schema))


def test_unevaluated_properties_sees_all_of() raises:
    """An in-place applicator's annotations reach the parent."""
    var schema = (
        '{"allOf":[{"properties":{"a":{}}},{"properties":{"b":{}}}],'
        '"unevaluatedProperties":false}'
    )
    assert_true(_valid('{"a":1,"b":2}', schema))
    assert_false(_valid('{"a":1,"b":2,"c":3}', schema))


def test_unevaluated_properties_sees_pattern_properties() raises:
    var schema = (
        '{"patternProperties":{"^x_":{}},"unevaluatedProperties":false}'
    )
    assert_true(_valid('{"x_1":1}', schema))
    assert_false(_valid('{"y":1}', schema))


def test_unevaluated_properties_sees_the_taken_branch() raises:
    var schema = (
        '{"properties":{"a":{}},'
        '"if":{"properties":{"a":{"const":1}},"required":["a"]},'
        '"then":{"properties":{"b":{}}},'
        '"else":{"properties":{"c":{}}},'
        '"unevaluatedProperties":false}'
    )
    assert_true(_valid('{"a":1,"b":2}', schema))
    assert_false(_valid('{"a":1,"c":2}', schema))
    assert_true(_valid('{"a":2,"c":3}', schema))
    assert_false(_valid('{"a":2,"b":3}', schema))


def test_unevaluated_properties_drops_a_failed_if() raises:
    """A failed `if` annotates nothing, so what it saw stays unevaluated."""
    var schema = (
        '{"if":{"properties":{"a":{"const":1}},"required":["a"]},'
        '"else":{"properties":{"c":{}}},'
        '"unevaluatedProperties":false}'
    )
    assert_false(_valid('{"a":2,"c":3}', schema))


def test_unevaluated_properties_as_a_schema() raises:
    var schema = (
        '{"properties":{"a":{}},"unevaluatedProperties":{"type":"string"}}'
    )
    assert_true(_valid('{"a":1,"b":"x"}', schema))
    assert_false(_valid('{"a":1,"b":2}', schema))


def test_unevaluated_items() raises:
    var schema = '{"prefixItems":[{}],"unevaluatedItems":false}'
    assert_true(_valid("[1]", schema))
    assert_false(_valid("[1,2]", schema))


def test_unevaluated_items_sees_contains() raises:
    var schema = '{"contains":{"type":"string"},"unevaluatedItems":false}'
    assert_true(_valid('["a"]', schema))
    assert_false(_valid('["a",1]', schema))


def test_unevaluated_items_after_items() raises:
    """`items` evaluates every element, so nothing is left over."""
    var schema = '{"items":{"type":"integer"},"unevaluatedItems":false}'
    assert_true(_valid("[1,2,3]", schema))


# ---------------------------------------------------------------------------
# format
# ---------------------------------------------------------------------------


def test_format_is_an_annotation_by_default() raises:
    assert_true(_valid('"not a date"', '{"format":"date"}'))


def test_format_annotations_are_collected() raises:
    var result = validate(loads('"not a date"'), loads('{"format":"date"}'))
    assert_true(result.valid)
    assert_equal(len(result.format_annotations), 1)
    assert_equal(result.format_annotations[0].format, "date")
    assert_false(result.format_annotations[0].matched)


def test_format_assertion_mode() raises:
    var schema = Schema.compile(loads('{"format":"date"}'), assert_format=True)
    assert_true(schema.is_valid(loads('"2020-02-29"')))
    assert_false(schema.is_valid(loads('"2021-02-29"')))
    assert_false(schema.is_valid(loads('"not a date"')))


def test_format_date_time_and_time() raises:
    var schema = Schema.compile(
        loads('{"format":"date-time"}'), assert_format=True
    )
    assert_true(schema.is_valid(loads('"1985-04-12T23:20:50.52Z"')))
    assert_true(schema.is_valid(loads('"1996-12-19T16:39:57-08:00"')))
    assert_false(schema.is_valid(loads('"1996-12-19 16:39:57Z"')))
    assert_false(schema.is_valid(loads('"1996-12-19T16:39:57"')))

    var time_schema = Schema.compile(
        loads('{"format":"time"}'), assert_format=True
    )
    assert_true(time_schema.is_valid(loads('"23:20:50.52Z"')))
    assert_false(time_schema.is_valid(loads('"25:20:50Z"')))


def test_format_network_attributes() raises:
    var host = Schema.compile(
        loads('{"format":"hostname"}'), assert_format=True
    )
    assert_true(host.is_valid(loads('"www.example.com"')))
    assert_false(host.is_valid(loads('"-bad.example.com"')))

    var v4 = Schema.compile(loads('{"format":"ipv4"}'), assert_format=True)
    assert_true(v4.is_valid(loads('"192.168.0.1"')))
    assert_false(v4.is_valid(loads('"192.168.0.256"')))
    assert_false(v4.is_valid(loads('"192.168.0"')))

    var v6 = Schema.compile(loads('{"format":"ipv6"}'), assert_format=True)
    assert_true(v6.is_valid(loads('"::1"')))
    assert_true(v6.is_valid(loads('"2001:db8::8a2e:370:7334"')))
    assert_true(v6.is_valid(loads('"::ffff:192.0.2.128"')))
    assert_false(v6.is_valid(loads('"2001:db8:::1"')))

    var mail = Schema.compile(loads('{"format":"email"}'), assert_format=True)
    assert_true(mail.is_valid(loads('"alice@example.com"')))
    assert_false(mail.is_valid(loads('"alice@@example.com"')))
    assert_false(mail.is_valid(loads('"alice"')))


def test_format_uri_uuid_regex_and_pointer() raises:
    var uri = Schema.compile(loads('{"format":"uri"}'), assert_format=True)
    assert_true(uri.is_valid(loads('"https://example.com/a?b#c"')))
    assert_false(uri.is_valid(loads('"/relative/path"')))

    var uuid = Schema.compile(loads('{"format":"uuid"}'), assert_format=True)
    assert_true(uuid.is_valid(loads('"f81d4fae-7dec-11d0-a765-00a0c91e6bf6"')))
    assert_false(uuid.is_valid(loads('"f81d4fae7dec11d0a76500a0c91e6bf6"')))

    var rx = Schema.compile(loads('{"format":"regex"}'), assert_format=True)
    assert_true(rx.is_valid(loads('"[a-z]+"')))
    assert_false(rx.is_valid(loads('"[a-z"')))

    var ptr = Schema.compile(
        loads('{"format":"json-pointer"}'), assert_format=True
    )
    assert_true(ptr.is_valid(loads('"/a~1b/0"')))
    assert_false(ptr.is_valid(loads('"a/b"')))


def test_unknown_format_never_fails() raises:
    var schema = Schema.compile(
        loads('{"format":"an-invented-format"}'), assert_format=True
    )
    assert_true(schema.is_valid(loads('"anything"')))


# ---------------------------------------------------------------------------
# The ten behaviours the previous version got wrong
# ---------------------------------------------------------------------------


def test_fixed_pattern_is_a_real_regex() raises:
    """`^[0-9]+$` used to match only the literal text `[0-9]+`."""
    assert_false(_valid('"[0-9]+"', '{"pattern":"^[0-9]+$"}'))
    assert_true(_valid('"7"', '{"pattern":"^[0-9]+$"}'))
    # Every unanchored pattern used to validate everything.
    assert_false(_valid('"abc"', '{"pattern":"[0-9]"}'))


def test_fixed_integer_accepts_a_zero_fraction() raises:
    """Section 6.1.1: a number with a zero fractional part is an integer."""
    assert_true(_valid("1.0", '{"type":"integer"}'))
    assert_true(_valid("-2.0", '{"type":"integer"}'))
    assert_false(_valid("1.5", '{"type":"integer"}'))


def test_fixed_length_counts_code_points() raises:
    """Five characters, seven bytes."""
    assert_true(_valid('"h\\u00e9ll\\u00f6"', '{"minLength":5,"maxLength":5}'))
    assert_false(_valid('"h\\u00e9ll\\u00f6"', '{"maxLength":4}'))


def test_fixed_equality_is_structural() raises:
    """Member order and the spelling of a number are not identity."""
    assert_true(_valid('{"a":1,"b":2}', '{"const":{"b":2,"a":1}}'))
    assert_true(_valid('{"a":1,"b":2}', '{"enum":[{"b":2,"a":1}]}'))
    assert_true(_valid("1.0", '{"const":1}'))
    assert_true(_valid("1", '{"enum":[1.0]}'))
    assert_false(
        _valid('[{"a":1,"b":2},{"b":2,"a":1}]', '{"uniqueItems":true}')
    )


def test_fixed_multiple_of_keeps_precision() raises:
    """The quotient used to be truncated through `Int` and overflow."""
    assert_true(_valid("9007199254740993", '{"multipleOf":1}'))
    assert_false(_valid("9007199254740993", '{"multipleOf":2}'))
    assert_true(_valid("1e30", '{"multipleOf":0.5}'))
    assert_true(_valid("0.0075", '{"multipleOf":0.0001}'))


def test_fixed_a_bad_schema_is_reported() raises:
    """A malformed schema used to read as an absent constraint."""
    _assert_schema_error('{"type":"strung"}')
    _assert_schema_error('{"minLength":"3"}')
    _assert_schema_error('{"required":"name"}')
    _assert_schema_error('{"properties":[]}')
    _assert_schema_error("[1,2,3]")
    _assert_schema_error('{"pattern":"(?=lookahead)"}')
    _assert_schema_error('{"minContains":2}')
    _assert_schema_error('{"items":[{"type":"integer"}]}')


def test_fixed_array_validation_does_not_stop_early() raises:
    """A throw while checking item 1 used to skip items 2 onward."""
    var result = validate(
        loads('["a","b","c"]'), loads('{"items":{"type":"integer"}}')
    )
    assert_false(result.valid)
    assert_equal(len(result.errors), 3)


def test_fixed_ref_is_not_ignored() raises:
    """`{"$ref": "#/$defs/Foo"}` used to validate everything."""
    var schema = '{"$defs":{"Foo":{"type":"string"}},"$ref":"#/$defs/Foo"}'
    assert_true(_valid('"x"', schema))
    assert_false(_valid("1", schema))
    # A reference that names nothing is an error rather than a pass.
    _assert_schema_error('{"$ref":"#/$defs/Missing"}')
    _assert_schema_error('{"$ref":"https://example.com/other.json"}')


def test_fixed_additional_properties_offsets_pattern_properties() raises:
    """Pairing the two keywords used to reject what the pattern matched."""
    var schema = (
        '{"patternProperties":{"^x_":{"type":"integer"}},'
        '"additionalProperties":false}'
    )
    assert_true(_valid('{"x_1":1,"x_2":2}', schema))
    assert_false(_valid('{"x_1":1,"y":2}', schema))


def test_fixed_error_paths_are_escaped_pointers() raises:
    """A member named `a/b` used to produce an ambiguous pointer."""
    var result = validate(
        loads('{"a/b":1}'), loads('{"properties":{"a/b":{"type":"string"}}}')
    )
    assert_false(result.valid)
    assert_equal(result.errors[0].path, "/a~1b")
    assert_equal(result.errors[0].keyword_location, "/properties/a~1b/type")

    var tilde = validate(
        loads('{"m~n":1}'), loads('{"properties":{"m~n":{"type":"string"}}}')
    )
    assert_equal(tilde.errors[0].path, "/m~0n")


def test_fixed_composition_keeps_its_causes() raises:
    """`anyOf` used to report one generic message and drop the rest."""
    var result = validate(
        loads("true"), loads('{"anyOf":[{"type":"string"},{"type":"integer"}]}')
    )
    assert_false(result.valid)
    # One summary plus the reason each of the two branches failed.
    assert_equal(len(result.errors), 3)
    assert_equal(result.errors[1].keyword_location, "/anyOf/0/type")
    assert_equal(result.errors[2].keyword_location, "/anyOf/1/type")


# ---------------------------------------------------------------------------
# Result surface
# ---------------------------------------------------------------------------


def test_validation_result_errors() raises:
    var result = validate(loads("{}"), loads('{"required":["name","age"]}'))
    assert_false(result.valid)
    assert_equal(len(result.errors), 2)


def test_validation_result_is_boolable() raises:
    assert_true(Bool(validate(loads("1"), loads('{"type":"integer"}'))))
    assert_false(Bool(validate(loads("1"), loads('{"type":"string"}'))))


def test_error_string_carries_the_instance_path() raises:
    var result = validate(
        loads('{"a":{"b":1}}'),
        loads('{"properties":{"a":{"properties":{"b":{"type":"string"}}}}}'),
    )
    assert_equal(result.errors[0].path, "/a/b")
    assert_true(String(result.errors[0]).startswith("/a/b: "))


def test_keyword_location_names_the_failing_keyword() raises:
    var result = validate(loads("3"), loads('{"minimum":5,"multipleOf":2}'))
    assert_false(result.valid)
    assert_equal(len(result.errors), 2)
    assert_equal(result.errors[0].keyword_location, "/minimum")
    assert_equal(result.errors[1].keyword_location, "/multipleOf")


# ---------------------------------------------------------------------------
# The compiled form
# ---------------------------------------------------------------------------


def test_compiled_schema_round_trip() raises:
    var schema = Schema.compile(loads('{"type":"integer","minimum":0}'))
    assert_true(schema.is_valid(loads("3")))
    assert_false(schema.is_valid(loads("-3")))
    assert_false(schema.is_valid(loads('"x"')))


def test_compiled_schema_is_reusable() raises:
    """The same compiled schema answers for many documents."""
    var schema = Schema.compile(loads('{"type":"string","pattern":"^a"}'))
    var documents = loads('["ab","ba","ac"]').array_items()
    var accepted = 0
    for i in range(len(documents)):
        if schema.is_valid(documents[i]):
            accepted += 1
    assert_equal(accepted, 2)


def test_compiled_schema_reports_a_bad_schema_at_compile_time() raises:
    var raised = False
    try:
        _ = Schema.compile(loads('{"type":42}'))
    except:
        raised = True
    assert_true(raised)


def main() raises:
    print("=" * 60)
    print("test_schema.mojo - JSON Schema draft 2020-12")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
