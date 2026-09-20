"""Runs the checked-in JSON Schema catalog against the validator.

The sibling runner `test_conformance.mojo` does this job for the
grammar catalogs. This one covers JSON Schema draft 2020-12: each case
carries a schema, an instance, the section of the specification the
rule comes from, and whether a conforming validator must call the
instance valid, must call it invalid, or must refuse the schema
outright. A failure prints the section URL, because the first move on a
failure is to read the rule rather than to argue with the case.

`schema-error` is the third outcome and the one that matters most here.
A validator that answers "valid" when it does not understand a schema
is worse than one with no opinion, because the caller cannot tell the
two apart: that is what a silently ignored `$ref` does, and it is why
the catalog asserts on refusal rather than leaving those cases out.

Cases marked `any` are reported and never fail: the specification
leaves them to the implementation, so what this library does with them
is information, not a verdict.

`_KNOWN_GAPS` lists the cases this release gets wrong. The assertion is
that the failing set is *exactly* that list, so a new failure breaks
the build and so does fixing one without removing its entry, which
keeps the list honest and shrinking.

Run it on its own:

    pixi run -e dev mojo run -I . tests/test_schema_conformance.mojo
"""

from std.collections import List
from std.pathlib import Path
from std.testing import assert_equal

from json import ValidationResult, Value, loads
from json.schema import Schema


# ---------------------------------------------------------------------------
# Cases this release gets wrong.
#
# Every entry is a bug, not a disagreement with the catalog, except
# where noted. Remove an entry in the same commit that fixes it.
# ---------------------------------------------------------------------------

# Nothing is known to be wrong. The assertion below is what keeps that
# true: a new failure fails the build rather than being absorbed here.
comptime _KNOWN_GAPS = List[String]()


@fieldwise_init
struct _Failure(Copyable):
    var id: String
    var expect: String
    var got: String
    var detail: String
    var section_url: String


def _has_key(v: Value, key: String) -> Bool:
    if not v.is_object():
        return False
    var keys = v.object_keys()
    for i in range(len(keys)):
        if keys[i] == key:
            return True
    return False


def _outcome(item: Value) raises -> Tuple[String, String]:
    """What the validator does with one case, and why.

    Returns the outcome name and, when the outcome is a refusal or a
    failed instance, the first thing the validator said about it. The
    detail is what turns a failing line from "disagrees" into something
    that can be acted on without a debugger.
    """
    var assert_format = False
    if _has_key(item, "assert_format"):
        assert_format = item["assert_format"].bool_value()

    var compiled: Schema
    try:
        compiled = Schema.compile(item["schema"], assert_format)
    except e:
        return ("schema-error", String(e))

    var result: ValidationResult
    try:
        result = compiled.validate(item["instance"])
    except e:
        # An evaluation that cannot finish is a refusal too: a `$ref`
        # cycle with no base case is only detectable once an instance
        # is being walked.
        return ("schema-error", String(e))

    if result.valid:
        return ("valid", "")
    var detail = String(result.errors[0])
    return ("invalid", detail)


def _run_catalog(path: String) raises -> Tuple[Int, Int, Int, List[_Failure]]:
    """Returns (asserted, informational, unexpected_passes, failures)."""
    var catalog = loads(Path(path).read_text())
    var cases = catalog["cases"].array_items()

    var asserted = 0
    var informational = 0
    var unexpected_passes = 0
    var failures = List[_Failure]()

    for i in range(len(cases)):
        ref item = cases[i]
        var id = item["id"].string_value()
        var expect = item["expect"].string_value()
        var outcome = _outcome(item)

        if expect == "any":
            informational += 1
            continue

        asserted += 1
        if outcome[0] == expect:
            if _is_known_gap(id):
                unexpected_passes += 1
                print("  fixed, remove from _KNOWN_GAPS:", id)
            continue
        if _is_known_gap(id):
            continue
        failures.append(
            _Failure(
                id,
                expect,
                outcome[0],
                outcome[1],
                item["section_url"].string_value(),
            )
        )

    return (asserted, informational, unexpected_passes, failures^)


def _is_known_gap(id: String) -> Bool:
    var ids = materialize[_KNOWN_GAPS]()
    for i in range(len(ids)):
        if ids[i] == id:
            return True
    return False


def _report(label: String, result: Tuple[Int, Int, Int, List[_Failure]]):
    ref failures = result[3]
    print(
        "  ",
        label + ":",
        result[0],
        "asserted,",
        result[1],
        "informational,",
        len(failures),
        "unexpected failures",
    )
    for i in range(len(failures)):
        print(
            "     ",
            failures[i].id,
            "-- expected",
            failures[i].expect + ",",
            "got " + failures[i].got,
            "--",
            failures[i].section_url,
        )
        if failures[i].detail != "":
            print("        ", failures[i].detail)


def run_catalogs() raises -> Int:
    """Report the catalog, then return the unexpected count."""
    var schema = _run_catalog("tests/conformance/jsonschema-2020-12.json")
    _report("JSON Schema 2020-12", schema)
    return len(schema[3]) + schema[2]


def test_every_keyword_is_covered() raises:
    """Assert the catalog exercises every keyword the module implements.

    A conformance file grows by accretion, and the failure mode is a
    keyword that was added to the validator and never added here, which
    no assertion on the cases themselves can notice.
    """
    var catalog = loads(
        Path("tests/conformance/jsonschema-2020-12.json").read_text()
    )
    var cases = catalog["cases"].array_items()

    var keywords = [
        "type",
        "enum",
        "const",
        "multipleOf",
        "maximum",
        "exclusiveMaximum",
        "minimum",
        "exclusiveMinimum",
        "maxLength",
        "minLength",
        "pattern",
        "maxItems",
        "minItems",
        "uniqueItems",
        "maxContains",
        "minContains",
        "maxProperties",
        "minProperties",
        "required",
        "dependentRequired",
        "format",
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "$anchor",
        "$comment",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
        "prefixItems",
        "items",
        "contains",
        "properties",
        "patternProperties",
        "additionalProperties",
        "propertyNames",
        "unevaluatedItems",
        "unevaluatedProperties",
    ]

    var missing = List[String]()
    for k in range(len(keywords)):
        var keyword = keywords[k]
        var found = False
        for i in range(len(cases)):
            if _mentions_keyword(cases[i]["schema"], keyword):
                found = True
                break
        if not found:
            missing.append(keyword)

    for i in range(len(missing)):
        print("   no catalog case uses", missing[i])
    print(
        "   keyword coverage:",
        len(keywords) - len(missing),
        "of",
        len(keywords),
    )
    assert_equal(len(missing), 0)


def _mentions_keyword(schema: Value, keyword: String) raises -> Bool:
    """Whether `keyword` appears as a member name anywhere in `schema`."""
    if schema.is_object():
        var members = schema.object_items()
        for i in range(len(members)):
            if members[i][0] == keyword:
                return True
            if _mentions_keyword(members[i][1], keyword):
                return True
        return False
    if schema.is_array():
        var items = schema.array_items()
        for i in range(len(items)):
            if _mentions_keyword(items[i], keyword):
                return True
    return False


def main() raises:
    print("Conformance catalogs:")
    var bad = run_catalogs()
    test_every_keyword_is_covered()
    print()
    assert_equal(bad, 0)
    print("All conformance catalogs matched expectations!")
