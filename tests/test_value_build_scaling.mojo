# Scaling guard for `Value` tree construction.
#
# This is a *complexity* test, not a speed test. It asserts that building
# a tree through the public API stays linear in the number of nodes.
#
# Why it exists: `Value.set` / `append` used to materialize the whole
# subtree into an `OwnedValue` and rebuild the document tape on every
# single call. Each mutation was O(subtree), so growing a tree was
# O(n^2). Building a 47 KB document took 19.5 ms, ~98% of it in
# construction rather than serialization, and the per-element cost grew
# with n (55 -> 75 -> 205 us/element at n = 1 / 10 / 100). Nothing in the
# suite caught it, because every correctness test used a handful of
# nodes where quadratic and linear look identical.
#
# The assertion compares the *per-element* cost at n=BIG against
# n=SMALL, rather than any absolute timing, since absolute numbers are
# not portable and CI machines are noisy. Linear construction holds that
# ratio near 1; the quadratic version grew it with n. See MAX_RATIO below
# for the measured bounds -- the spread between SMALL and BIG matters,
# because a large linear constant per document masks the quadratic term
# until n gets big enough.

from std.testing import assert_true, TestSuite
from std.time import perf_counter_ns

from json import dumps, loads, Value


comptime SMALL = 10
comptime BIG = 400

# Ratio of per-element cost at BIG vs SMALL that we tolerate.
#
# These bounds are measured, not guessed. Against the pre-fix
# implementation the observed ratios were:
#
#     n=10 -> n=100   2.84
#     n=10 -> n=200   5.00
#     n=10 -> n=400   9.26
#
# The quadratic term is masked at small n by a large per-document linear
# constant, which is why SMALL/BIG have to be this far apart: a limit of
# 6.0 at n=200 would have let the original regression through. At n=400
# the fixed implementation measures ~1.1-1.3, so 3.0 sits with wide
# margin on both sides -- roughly 2.5x headroom over noise, and a 3x gap
# below the broken behaviour.
comptime MAX_RATIO = 3.0

# Repeats of each measurement; we keep the best to shed scheduler noise.
comptime REPS = 5


def _build_doc(idx: Int) raises -> Value:
    """One `document`-shaped node: a few scalars, a submap, 8 children."""
    var o = Value.object()
    o.set("id", Value("doc-" + String(idx)))
    o.set("status", Value(Int64(idx % 5)))
    var meta = Value.object()
    meta.set("region", Value("us-east-1"))
    meta.set("version", Value(Int64(idx % 3)))
    o.set("meta", meta)
    var items = Value.array()
    for i in range(8):
        var it = Value.object()
        it.set("sku", Value("sku-" + String(i)))
        it.set("qty", Value(Int64(i)))
        it.set("price_minor", Value(Int64(i * 251)))
        items.append(it)
    o.set("items", items)
    return o^


def _build_n(n: Int) raises -> Value:
    var items = Value.array()
    for i in range(n):
        items.append(_build_doc(i))
    var wrap = Value.object()
    wrap.set("items", items)
    return wrap^


def _best_build_ns(n: Int) raises -> Int:
    """Best-of-REPS wall time to build a tree of `n` documents."""
    _ = _build_n(n)  # warm the allocator
    var best = 0
    for r in range(REPS):
        var t0 = Int(perf_counter_ns())
        var v = _build_n(n)
        var t1 = Int(perf_counter_ns())
        # Keep the tree observable so the build cannot be optimized away.
        _ = v.object_count()
        if r == 0 or (t1 - t0) < best:
            best = t1 - t0
    return best


def test_build_scales_linearly() raises:
    """Per-element build cost must not grow with the element count."""
    var small_ns = _best_build_ns(SMALL)
    var big_ns = _best_build_ns(BIG)

    var per_small = Float64(small_ns) / Float64(SMALL)
    var per_big = Float64(big_ns) / Float64(BIG)
    var ratio = per_big / per_small

    print("      n=", SMALL, " total=", small_ns, "ns  per-elem=", per_small)
    print("      n=", BIG, " total=", big_ns, "ns  per-elem=", per_big)
    print("      ratio=", ratio, " (limit ", MAX_RATIO, ")")

    assert_true(
        ratio < MAX_RATIO,
        String("Value construction is not scaling linearly: per-element cost")
        + " at n="
        + String(BIG)
        + " is "
        + String(ratio)
        + "x the cost at n="
        + String(SMALL)
        + " (limit "
        + String(MAX_RATIO)
        + "). set/append most likely rebuild the whole subtree again --"
        + " see json/value/value.mojo.",
    )


def test_legacy_loads_build_scales_linearly() raises:
    """Same guard for the `loads("{}")` + `set` shape.

    Callers who predate `Value.object()` build trees this way, and it was
    the shape that exhibited the original blowup. It must scale too.
    """
    var small = loads("[]")
    var t0 = Int(perf_counter_ns())
    for i in range(SMALL):
        small.append(_build_doc(i))
    var t1 = Int(perf_counter_ns())

    var big = loads("[]")
    var t2 = Int(perf_counter_ns())
    for i in range(BIG):
        big.append(_build_doc(i))
    var t3 = Int(perf_counter_ns())

    var per_small = Float64(t1 - t0) / Float64(SMALL)
    var per_big = Float64(t3 - t2) / Float64(BIG)
    var ratio = per_big / per_small
    print("      legacy ratio=", ratio, " (limit ", MAX_RATIO, ")")

    assert_true(
        ratio < MAX_RATIO,
        String("Appending into a parsed array is not linear: ")
        + String(ratio)
        + "x per-element cost from n="
        + String(SMALL)
        + " to n="
        + String(BIG),
    )


def test_build_output_is_correct_at_scale() raises:
    """Linearity is worthless if the tree is wrong -- check the payload."""
    var v = _build_n(BIG)
    var text = dumps(v)
    var back = loads(text)
    assert_true(back["items"].array_count() == BIG)
    assert_true(back["items"][BIG - 1]["items"].array_count() == 8)
    assert_true(
        back["items"][BIG - 1]["id"].string_value() == "doc-" + String(BIG - 1)
    )
    # Re-serializing the parsed copy must reproduce the same bytes.
    assert_true(dumps(back) == text)


def main() raises:
    print("=" * 60)
    print("test_value_build_scaling.mojo")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
