"""Runs the JSON Pointer, JSON Patch and JSON Merge Patch catalogs.

The sibling runner `test_conformance.mojo` does the same job for the
grammar catalogs. This one covers the three specifications layered on
top of the grammar: RFC 6901 addresses a location inside a document,
RFC 6902 edits a document through a list of operations, and RFC 7396
edits it by overlaying a sparse copy of it.

Unlike the grammar runner, this file is deliberately not wired into
`tests-cpu`. These three specifications are still being brought up, so
asserting on the catalogs would mean pinning a list of known gaps that
moves under the next commit. The output is the point instead: every
failure is sorted into one of four root causes, printed under that
heading, and carries both the section URL and whatever the library
actually did. Wiring this into CI is the last step of closing the gaps,
not the first.

The four root causes are:

- accepted-invalid: the library returned a value where the RFC requires
  an error. This is the dangerous one, because the caller gets a
  plausible-looking wrong answer with no signal.
- rejected-valid: the library raised where the RFC requires success.
  Loud, and therefore the cheapest kind to find.
- wrong-result: the library succeeded and produced the wrong document.
- not-atomic: a failing patch left the caller's document modified.

A fifth outcome has no category because it cannot be caught: a bounds
assertion inside the library aborts the process rather than raising, so
the report stops at the catalog it was in the middle of and prints
nothing for it. Set `_TRACE` to True to print each case id before it
runs, which names the case that aborted.

Run it on its own:

    pixi run -e dev mojo run -I . tests/test_pointer_patch_conformance.mojo
"""

from std.collections import List
from std.pathlib import Path

from json import Value, apply_patch, create_merge_patch, loads, merge_patch


# Flip to True to print every case id as it starts. The only reason to
# want that is to find the case that aborts the process, since an abort
# takes the report with it.
comptime _TRACE = False

comptime _ACCEPTED_INVALID = "accepted-invalid"
comptime _REJECTED_VALID = "rejected-valid"
comptime _WRONG_RESULT = "wrong-result"
comptime _NOT_ATOMIC = "not-atomic"

# Printed in this order, worst first, so the head of the report is the
# set of cases where a caller gets a wrong answer without an exception.
comptime _CATEGORIES = [
    _ACCEPTED_INVALID,
    _WRONG_RESULT,
    _NOT_ATOMIC,
    _REJECTED_VALID,
]

comptime _CATEGORY_BLURBS = [
    "returned a value where the RFC requires an error",
    "succeeded but produced the wrong document",
    "left the caller's document modified after a failed patch",
    "raised where the RFC requires success",
]


@fieldwise_init
struct _Failure(Copyable):
    """One case the library gets wrong, with what it did instead."""

    var id: String
    var category: String
    var title: String
    var observed: String
    var section_url: String


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------


def _has_key(v: Value, key: String) -> Bool:
    if not v.is_object():
        return False
    var keys = v.object_keys()
    for i in range(len(keys)):
        if keys[i] == key:
            return True
    return False


def _as_float(v: Value) raises -> Float64:
    """A number as a Float64, whichever way the tape stored it."""
    if v.is_float():
        return v.float_value()
    return Float64(v.int_value())


def _deep_equal(a: Value, b: Value) raises -> Bool:
    """JSON equality as RFC 6902 section 4.6 defines it.

    `Value.__eq__` compares serialized text, which makes two objects
    that differ only in member order unequal and makes `1` unequal to
    `1.0`. RFC 8259 gives object members no order and RFC 6902 says
    numbers compare by value, so the catalogs need this comparison
    rather than that one.
    """
    if a.is_null() or b.is_null():
        return a.is_null() and b.is_null()
    if a.is_bool() or b.is_bool():
        return a.is_bool() and b.is_bool() and a.bool_value() == b.bool_value()
    if a.is_string() or b.is_string():
        if not (a.is_string() and b.is_string()):
            return False
        return a.string_value() == b.string_value()
    if (a.is_int() or a.is_float()) and (b.is_int() or b.is_float()):
        return _as_float(a) == _as_float(b)
    if a.is_array() and b.is_array():
        var xs = a.array_items()
        var ys = b.array_items()
        if len(xs) != len(ys):
            return False
        for i in range(len(xs)):
            if not _deep_equal(xs[i], ys[i]):
                return False
        return True
    if a.is_object() and b.is_object():
        var ka = a.object_keys()
        var kb = b.object_keys()
        if len(ka) != len(kb):
            return False
        for i in range(len(ka)):
            if not _has_key(b, ka[i]):
                return False
            if not _deep_equal(a[ka[i]], b[ka[i]]):
                return False
        return True
    return False


def _brief(v: Value) raises -> String:
    """A one-line rendering of a value, clipped so a report stays readable."""
    var text = String(v)
    if text.byte_length() <= 72:
        return text^
    # Back off to a byte that starts a code point, so clipping a value
    # that happens to hold non-ASCII does not emit broken UTF-8.
    var cut = 69
    var raw = text.as_bytes()
    while cut > 0 and (raw[cut] & 0xC0) == 0x80:
        cut -= 1
    return String(String(unsafe_from_utf8=raw[:cut])) + "..."


def _trace(id: String):
    if materialize[_TRACE]():
        print("      running", id)


def _load_cases(path: String) raises -> Value:
    var catalog = loads(Path(path).read_text())
    return catalog["cases"]


# ---------------------------------------------------------------------------
# RFC 6901 -- JSON Pointer, exercised through `Value.at`
# ---------------------------------------------------------------------------


def _run_pointer_catalog(
    path: String,
) raises -> Tuple[Int, Int, List[_Failure]]:
    """Returns (asserted, informational, failures)."""
    var cases = _load_cases(path).array_items()
    var asserted = 0
    var informational = 0
    var failures = List[_Failure]()

    for i in range(len(cases)):
        ref item = cases[i]
        var id = item["id"].string_value()
        var title = item["title"].string_value()
        var url = item["section_url"].string_value()
        var expect = item["expect"].string_value()
        var pointer = item["pointer"].string_value()

        _trace(id)
        asserted += 1

        var resolved = False
        var matched = False
        var shown = String()
        try:
            var got = item["document"].at(pointer)
            resolved = True
            shown = _brief(got)
            if expect == "resolve":
                matched = _deep_equal(got, item["result"])
        except e:
            if not resolved:
                shown = String(e)

        if expect == "resolve":
            if not resolved:
                failures.append(
                    _Failure(id, _REJECTED_VALID, title, "raised " + shown, url)
                )
            elif not matched:
                failures.append(
                    _Failure(
                        id,
                        _WRONG_RESULT,
                        title,
                        "got " + shown + ", wanted " + _brief(item["result"]),
                        url,
                    )
                )
        else:
            if resolved:
                failures.append(
                    _Failure(
                        id,
                        _ACCEPTED_INVALID,
                        title,
                        "resolved to " + shown,
                        url,
                    )
                )

    return (asserted, informational, failures^)


# ---------------------------------------------------------------------------
# RFC 6902 -- JSON Patch, exercised through `apply_patch`
# ---------------------------------------------------------------------------


def _run_patch_catalog(path: String) raises -> Tuple[Int, Int, List[_Failure]]:
    """Returns (asserted, informational, failures)."""
    var cases = _load_cases(path).array_items()
    var asserted = 0
    var informational = 0
    var failures = List[_Failure]()

    for i in range(len(cases)):
        ref item = cases[i]
        var id = item["id"].string_value()
        var title = item["title"].string_value()
        var url = item["section_url"].string_value()
        var expect = item["expect"].string_value()

        _trace(id)
        asserted += 1

        # The document is copied first so that a patch that edits its
        # input in place has something of ours to damage. Comparing the
        # copy against the catalog afterwards is what detects that.
        var document = item["document"].copy()

        var applied = False
        var matched = False
        var shown = String()
        try:
            var patch: Value
            if _has_key(item, "patch_text"):
                patch = loads(item["patch_text"].string_value())
            else:
                patch = item["patch"]
            var got = apply_patch(document, patch)
            applied = True
            shown = _brief(got)
            if expect == "apply":
                matched = _deep_equal(got, item["result"])
        except e:
            if not applied:
                shown = String(e)

        if expect == "apply":
            if not applied:
                failures.append(
                    _Failure(id, _REJECTED_VALID, title, "raised " + shown, url)
                )
            elif not matched:
                failures.append(
                    _Failure(
                        id,
                        _WRONG_RESULT,
                        title,
                        "got " + shown + ", wanted " + _brief(item["result"]),
                        url,
                    )
                )
        else:
            if applied:
                failures.append(
                    _Failure(
                        id, _ACCEPTED_INVALID, title, "produced " + shown, url
                    )
                )

        # Atomicity is a separate claim from the verdict above: a patch
        # may correctly report an error and still have scribbled on the
        # document on its way there.
        if _has_key(item, "check_unchanged"):
            asserted += 1
            if not _deep_equal(document, item["document"]):
                failures.append(
                    _Failure(
                        id,
                        _NOT_ATOMIC,
                        title,
                        "document became " + _brief(document),
                        url,
                    )
                )

    return (asserted, informational, failures^)


# ---------------------------------------------------------------------------
# RFC 7396 -- JSON Merge Patch, through `merge_patch` / `create_merge_patch`
# ---------------------------------------------------------------------------


def _run_merge_catalog(path: String) raises -> Tuple[Int, Int, List[_Failure]]:
    """Returns (asserted, informational, failures).

    A `diff` case asserts only the round trip, that applying the
    generated patch to the source yields the target. RFC 7396 section 3
    describes what a generator should emit without making any one
    output normative, so the comparison against the catalog's own
    `expected_patch` is reported as informational instead.
    """
    var cases = _load_cases(path).array_items()
    var asserted = 0
    var informational = 0
    var failures = List[_Failure]()

    for i in range(len(cases)):
        ref item = cases[i]
        var id = item["id"].string_value()
        var title = item["title"].string_value()
        var url = item["section_url"].string_value()
        var kind = item["kind"].string_value()

        _trace(id)
        if kind == "merge":
            asserted += 1
            var applied = False
            var matched = False
            var shown = String()
            try:
                var got = merge_patch(item["original"], item["patch"])
                applied = True
                shown = _brief(got)
                matched = _deep_equal(got, item["result"])
            except e:
                if not applied:
                    shown = String(e)

            if not applied:
                failures.append(
                    _Failure(id, _REJECTED_VALID, title, "raised " + shown, url)
                )
            elif not matched:
                failures.append(
                    _Failure(
                        id,
                        _WRONG_RESULT,
                        title,
                        "got " + shown + ", wanted " + _brief(item["result"]),
                        url,
                    )
                )
            continue

        # A diff case.
        asserted += 1
        informational += 1
        var built = False
        var round_trips = False
        var exact = False
        var shown = String()
        try:
            var patch = create_merge_patch(item["source"], item["target"])
            built = True
            shown = _brief(patch)
            exact = _deep_equal(patch, item["expected_patch"])
            round_trips = _deep_equal(
                merge_patch(item["source"], patch), item["target"]
            )
        except e:
            if not built:
                shown = String(e)

        if not built:
            failures.append(
                _Failure(id, _REJECTED_VALID, title, "raised " + shown, url)
            )
        elif not round_trips:
            failures.append(
                _Failure(
                    id,
                    _WRONG_RESULT,
                    title,
                    "patch " + shown + " does not transform source to target",
                    url,
                )
            )
        elif not exact:
            print(
                "      note:",
                id,
                "round-trips but emits",
                shown,
                "rather than",
                _brief(item["expected_patch"]),
            )

    return (asserted, informational, failures^)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _report(label: String, result: Tuple[Int, Int, List[_Failure]]) raises:
    ref failures = result[2]
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

    var categories = materialize[_CATEGORIES]()
    var blurbs = materialize[_CATEGORY_BLURBS]()
    for c in range(len(categories)):
        var count = 0
        for i in range(len(failures)):
            if failures[i].category == categories[c]:
                count += 1
        if count == 0:
            continue
        print(
            "      ",
            String(categories[c]) + ":",
            count,
            "--",
            String(blurbs[c]),
        )
        for i in range(len(failures)):
            if failures[i].category != categories[c]:
                continue
            print("         ", failures[i].id, "--", failures[i].observed)
            print("            ", failures[i].section_url)


def main() raises:
    print("Pointer and patch conformance catalogs:")

    print("   RFC 6901 (JSON Pointer) ...")
    var pointer = _run_pointer_catalog("tests/conformance/rfc6901-pointer.json")
    _report("RFC 6901", pointer)

    print("   RFC 6902 (JSON Patch) ...")
    var patch = _run_patch_catalog("tests/conformance/rfc6902-patch.json")
    _report("RFC 6902", patch)

    print("   RFC 7396 (JSON Merge Patch) ...")
    var merge = _run_merge_catalog("tests/conformance/rfc7396-merge-patch.json")
    _report("RFC 7396", merge)

    var bad = len(pointer[2]) + len(patch[2]) + len(merge[2])
    var total = pointer[0] + patch[0] + merge[0]
    print()
    print("  ", total - bad, "of", total, "asserted checks match the RFC text.")
    if bad != 0:
        # This file is not part of `tests-cpu`, so it reports rather
        # than fails. Turning the count into an assertion is the last
        # step of closing the gaps, not the first.
        print("  ", bad, "checks do not. See the groupings above.")
