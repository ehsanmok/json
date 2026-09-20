"""Runs the checked-in specification catalogs against `loads`.

Each catalog case carries the RFC section it comes from and whether a
conforming parser must accept the input, must reject it, or may do
either. A failure prints the section URL, because the first move on a
failure is to read the rule rather than to argue with the case.

Cases marked `any` are reported and never fail: the standard leaves
them to the implementation, so what we do with them is information,
not a verdict.

`_KNOWN_GAPS` lists the cases this release gets wrong. The assertion is
that the failing set is *exactly* that list, so a new failure breaks
the build and so does fixing one without removing its entry -- which
keeps the list honest and shrinking.
"""

from std.collections import Dict, List
from std.pathlib import Path
from std.testing import assert_equal, assert_true

from json import ParserConfig, Value, loads


# ---------------------------------------------------------------------------
# Cases this release is known to get wrong.
#
# Every entry is a bug, not a disagreement with the catalog, except
# where noted. Remove an entry in the same commit that fixes it. The
# list is empty, and the assertion below is what keeps it that way:
# a new failure fails the build rather than being absorbed here.
# ---------------------------------------------------------------------------

comptime _KNOWN_GAPS = List[String]()

# Cases we fail on purpose, with the reason. These are all SHOULD-level
# and the catalog says so in its own notes; they are not bugs and this
# list is not expected to shrink.
comptime _DELIBERATE = [
    # RFC 8259 section 8.2 does not forbid unpaired surrogates -- the
    # catalog records the case at SHOULD NOT for interchange, and its
    # note says as much. An unpaired escape becomes U+FFFD here, which
    # is what the section describes software doing, and I-JSON mode
    # rejects it outright. Rejecting it by default would refuse
    # documents the grammar allows.
    "json-8259-lone-surrogate",
]


@fieldwise_init
struct _Failure(Copyable):
    var id: String
    var expect: String
    var got: String
    var section_url: String


def _hex_value(c: UInt8) raises -> Int:
    var v = Int(c)
    if v >= ord("0") and v <= ord("9"):
        return v - ord("0")
    if v >= ord("a") and v <= ord("f"):
        return v - ord("a") + 10
    if v >= ord("A") and v <= ord("F"):
        return v - ord("A") + 10
    raise Error("conformance: bad hex digit in case input")


def _case_bytes(item: Value) raises -> String:
    """The case input as raw bytes, whatever encoding it is stored in.

    A `hex` case is one whose bytes are not valid UTF-8 (or not text at
    all), which is exactly what the catalog needs it for, so it cannot
    be stored as a JSON string. The parser works on bytes, so the
    decoded bytes are handed over without a validity claim.
    """
    var text = item["input"].string_value()
    var encoding = "utf-8"
    if _has_key(item, "input_encoding"):
        encoding = item["input_encoding"].string_value()
    if encoding != "hex":
        return text^

    var src = text.as_bytes()
    if len(src) % 2 != 0:
        raise Error("conformance: odd-length hex input")
    var out = List[UInt8](capacity=len(src) // 2)
    var i = 0
    while i < len(src):
        out.append(UInt8(_hex_value(src[i]) * 16 + _hex_value(src[i + 1])))
        i += 2
    return String(unsafe_from_utf8=out^)


def _has_key(v: Value, key: String) -> Bool:
    if not v.is_object():
        return False
    var keys = v.object_keys()
    for i in range(len(keys)):
        if keys[i] == key:
            return True
    return False


def _accepts(payload: String, config: ParserConfig) -> Bool:
    try:
        var parsed = loads(payload, config)
        _ = parsed.is_null()
        return True
    except:
        return False


def _is_known_gap(id: String) -> Bool:
    var ids = materialize[_KNOWN_GAPS]()
    for i in range(len(ids)):
        if ids[i] == id:
            return True
    return False


def _is_deliberate(id: String) -> Bool:
    var ids = materialize[_DELIBERATE]()
    for i in range(len(ids)):
        if ids[i] == id:
            return True
    return False


def _run_catalog(
    path: String, config: ParserConfig
) raises -> Tuple[Int, Int, Int, List[_Failure]]:
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
        var accepted = _accepts(_case_bytes(item), config)

        if expect == "any":
            informational += 1
            continue

        asserted += 1
        var wanted_accept = expect == "accept"
        if accepted == wanted_accept:
            if _is_deliberate(id):
                unexpected_passes += 1
                print("  no longer diverges, remove from _DELIBERATE:", id)
            elif _is_known_gap(id):
                unexpected_passes += 1
                print(
                    "  fixed, remove from _KNOWN_GAPS:",
                    id,
                )
            continue
        if _is_known_gap(id) or _is_deliberate(id):
            continue
        failures.append(
            _Failure(
                id,
                expect,
                "accepted" if accepted else "rejected",
                item["section_url"].string_value(),
            )
        )

    return (asserted, informational, unexpected_passes, failures^)


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
            failures[i].got,
            "--",
            failures[i].section_url,
        )


def run_catalogs() raises -> Int:
    """Report every catalog, then return the total unexpected count.

    Reporting all of them before asserting matters: stopping at the
    first catalog with a failure hides whatever the others would have
    said, and the list of what is wrong is the useful output here.
    """
    var bad = 0

    var core = _run_catalog("tests/conformance/rfc8259.json", ParserConfig())
    _report("RFC 8259", core)
    bad += len(core[3]) + core[2]

    # I-JSON is RFC 8259 narrowed, so its catalog runs under an I-JSON
    # parser configuration. The narrowing rules used to sit in
    # `_KNOWN_GAPS` because there was no such mode.
    var ijson = _run_catalog(
        "tests/conformance/rfc7493-ijson.json", ParserConfig.interoperable()
    )
    _report("RFC 7493", ijson)
    bad += len(ijson[3]) + ijson[2]

    return bad


def test_decoded_values() raises:
    """Cases that pin the parsed value, not just acceptance."""
    var catalog = loads(Path("tests/conformance/rfc8259.json").read_text())
    var cases = catalog["cases"].array_items()
    var checked = 0
    for i in range(len(cases)):
        ref item = cases[i]
        if not _has_key(item, "decoded"):
            continue
        if item["expect"].string_value() != "accept":
            continue
        if _is_known_gap(item["id"].string_value()):
            continue
        var parsed = loads(_case_bytes(item))
        assert_true(
            parsed == item["decoded"],
            "decoded mismatch for " + item["id"].string_value(),
        )
        checked += 1
    print("   decoded values:", checked, "cases")


def main() raises:
    print("Conformance catalogs:")
    var bad = run_catalogs()
    test_decoded_values()
    print()
    assert_equal(bad, 0)
    print("All conformance catalogs matched expectations!")
