# Value construction (write-path) benchmark.
#
# Mirrors the `document` fixture shape used by `bench_serde.mojo`
# (children=8, fields_per_child=3, max_depth=2), because that is the
# shape a Value tree is most often built into by hand.
#
# The harness had no way to construct an empty container, so it called
# `loads("{}")` / `loads("[]")` per node and then grew the tree with
# `set` / `append`. Both shapes are timed here:
#
#   legacy   -- loads("{}") per node, then set/append   (what the suite ran)
#   factory  -- Value.object() / Value.array()          (the intended path)
#   moved    -- same, but transferring containers with `^`
#
# The gap between `factory` and `moved` is the cost of deep-copying a
# subtree on insert. `set`/`append` have both borrowed and `var`
# overloads; passing a named local takes the borrowed one and copies, so
# the value stays usable, while `items^` hands the subtree over. Worth
# knowing when building large trees.
#
# Reported per n: build-only, dumps-only, and the sum. The scaling column is
# what matters: per-element cost must stay flat as n grows. If it climbs
# linearly, tree construction has gone quadratic again.
#
# Run: pixi run -e dev bench-build   (or: mojo -I . benchmark/mojo/bench_build.mojo)

from std.time import perf_counter_ns

from json import Value, dumps, loads


comptime CHILDREN = 8
comptime WARMUP = 3
comptime ITERS = 20


# ---------------------------------------------------------------------------
# Fixture (mirrors the suite's `document` type)
# ---------------------------------------------------------------------------


struct DocItem(Copyable):
    var sku: String
    var qty: Int32
    var price_minor: Int64

    def __init__(out self, idx: Int):
        self.sku = "sku-" + String(idx % 997)
        self.qty = Int32(idx % 13)
        self.price_minor = Int64(idx * 251 % 100000)


struct Doc(Copyable):
    var id: String
    var status: Int32
    var region: String
    var version: Int32
    var items: List[DocItem]

    def __init__(out self, idx: Int):
        self.id = "doc-" + String(idx)
        self.status = Int32(idx % 5)
        self.region = "us-east-1"
        self.version = Int32(idx % 3)
        self.items = List[DocItem](capacity=CHILDREN)
        for i in range(CHILDREN):
            self.items.append(DocItem(idx * CHILDREN + i))


def make_docs(n: Int) -> List[Doc]:
    var out = List[Doc](capacity=n)
    for i in range(n):
        out.append(Doc(i))
    return out^


# ---------------------------------------------------------------------------
# Build path A: `loads("{}")` per node (what the benchmark suite ran)
# ---------------------------------------------------------------------------


def _legacy_doc(d: Doc) raises -> Value:
    var o = loads("{}")
    o.set("id", Value(d.id))
    o.set("status", Value(Int(d.status)))
    var meta = loads("{}")
    meta.set("region", Value(d.region))
    meta.set("version", Value(Int(d.version)))
    o.set("meta", meta)
    var items = loads("[]")
    for i in range(len(d.items)):
        var it = loads("{}")
        it.set("sku", Value(d.items[i].sku))
        it.set("qty", Value(Int(d.items[i].qty)))
        it.set("price_minor", Value(d.items[i].price_minor))
        items.append(it)
    o.set("items", items)
    return o^


def build_legacy(docs: List[Doc]) raises -> Value:
    if len(docs) == 1:
        return _legacy_doc(docs[0])
    var items = loads("[]")
    for i in range(len(docs)):
        items.append(_legacy_doc(docs[i]))
    var wrap = loads("{}")
    wrap.set("items", items)
    return wrap^


# ---------------------------------------------------------------------------
# Build path B: the construction factories
# ---------------------------------------------------------------------------


def _factory_doc(d: Doc) raises -> Value:
    var o = Value.object()
    o.set("id", Value(d.id))
    o.set("status", Value(Int(d.status)))
    var meta = Value.object()
    meta.set("region", Value(d.region))
    meta.set("version", Value(Int(d.version)))
    o.set("meta", meta)
    var items = Value.array()
    for i in range(len(d.items)):
        var it = Value.object()
        it.set("sku", Value(d.items[i].sku))
        it.set("qty", Value(Int(d.items[i].qty)))
        it.set("price_minor", Value(d.items[i].price_minor))
        items.append(it)
    o.set("items", items)
    return o^


def build_factory(docs: List[Doc]) raises -> Value:
    if len(docs) == 1:
        return _factory_doc(docs[0])
    var items = Value.array()
    for i in range(len(docs)):
        items.append(_factory_doc(docs[i]))
    var wrap = Value.object()
    wrap.set("items", items)
    return wrap^


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _pad(s: String, width: Int, left: Bool) -> String:
    var out = s
    while out.byte_length() < width:
        if left:
            out = " " + out
        else:
            out += " "
    return out^


def _fmt_us(ns: Float64) -> String:
    """Nanoseconds -> microseconds, three decimals."""
    var us = ns / 1000.0
    var scaled = Int64(us * 1000.0 + 0.5)
    var whole = scaled // 1000
    var frac = scaled % 1000
    var frac_s = String(frac)
    while frac_s.byte_length() < 3:
        frac_s = "0" + frac_s
    return String(whole) + "." + frac_s


def _row(label: String, n: Int, build_ns: Int, dumps_ns: Int, size: Int):
    var total = build_ns + dumps_ns
    print(
        "  ",
        _pad(label, 9, False),
        _pad(String(n), 5, True),
        _pad(_fmt_us(Float64(build_ns)), 12, True),
        _pad(_fmt_us(Float64(dumps_ns)), 12, True),
        _pad(_fmt_us(Float64(total)), 12, True),
        _pad(String(size), 9, True),
        _pad(_fmt_us(Float64(total) / Float64(n)), 12, True),
    )


def bench_legacy(n: Int) raises:
    var docs = make_docs(n)
    for _ in range(WARMUP):
        _ = dumps(build_legacy(docs))

    var best_build = 0
    var best_dumps = 0
    var size = 0
    for i in range(ITERS):
        var t0 = Int(perf_counter_ns())
        var v = build_legacy(docs)
        var t1 = Int(perf_counter_ns())
        var s = dumps(v)
        var t2 = Int(perf_counter_ns())
        size = s.byte_length()
        if i == 0 or (t1 - t0) < best_build:
            best_build = t1 - t0
        if i == 0 or (t2 - t1) < best_dumps:
            best_dumps = t2 - t1
    _row("legacy", n, best_build, best_dumps, size)


def bench_factory(n: Int) raises:
    var docs = make_docs(n)
    for _ in range(WARMUP):
        _ = dumps(build_factory(docs))

    var best_build = 0
    var best_dumps = 0
    var size = 0
    for i in range(ITERS):
        var t0 = Int(perf_counter_ns())
        var v = build_factory(docs)
        var t1 = Int(perf_counter_ns())
        var s = dumps(v)
        var t2 = Int(perf_counter_ns())
        size = s.byte_length()
        if i == 0 or (t1 - t0) < best_build:
            best_build = t1 - t0
        if i == 0 or (t2 - t1) < best_dumps:
            best_dumps = t2 - t1
    _row("factory", n, best_build, best_dumps, size)


# ---------------------------------------------------------------------------
# Build path C: factories, transferring ownership of each container
# ---------------------------------------------------------------------------


def _moved_doc(d: Doc) raises -> Value:
    var o = Value.object()
    o.set("id", Value(d.id))
    o.set("status", Value(Int(d.status)))
    var meta = Value.object()
    meta.set("region", Value(d.region))
    meta.set("version", Value(Int(d.version)))
    o.set("meta", meta^)
    var items = Value.array()
    for i in range(len(d.items)):
        var it = Value.object()
        it.set("sku", Value(d.items[i].sku))
        it.set("qty", Value(Int(d.items[i].qty)))
        it.set("price_minor", Value(d.items[i].price_minor))
        items.append(it^)
    o.set("items", items^)
    return o^


def build_moved(docs: List[Doc]) raises -> Value:
    if len(docs) == 1:
        return _moved_doc(docs[0])
    var items = Value.array()
    for i in range(len(docs)):
        items.append(_moved_doc(docs[i]))
    var wrap = Value.object()
    wrap.set("items", items^)
    return wrap^


def bench_moved(n: Int) raises:
    var docs = make_docs(n)
    for _ in range(WARMUP):
        _ = dumps(build_moved(docs))

    var best_build = 0
    var best_dumps = 0
    var size = 0
    for i in range(ITERS):
        var t0 = Int(perf_counter_ns())
        var v = build_moved(docs)
        var t1 = Int(perf_counter_ns())
        var s = dumps(v)
        var t2 = Int(perf_counter_ns())
        size = s.byte_length()
        if i == 0 or (t1 - t0) < best_build:
            best_build = t1 - t0
        if i == 0 or (t2 - t1) < best_dumps:
            best_dumps = t2 - t1
    _row("moved", n, best_build, best_dumps, size)


def bench_parse(n: Int) raises:
    """Read path, for contrast: parse the same payload back."""
    var docs = make_docs(n)
    var payload = dumps(build_factory(docs))
    for _ in range(WARMUP):
        _ = loads(payload)

    var best = 0
    for i in range(ITERS):
        var t0 = Int(perf_counter_ns())
        var v = loads(payload)
        var t1 = Int(perf_counter_ns())
        _ = v.is_object()
        if i == 0 or (t1 - t0) < best:
            best = t1 - t0
    _row("loads", n, 0, best, payload.byte_length())


def main() raises:
    print(
        "Value construction benchmark -- document fixture, children=", CHILDREN
    )
    print("best-of", ITERS, "after", WARMUP, "warmup; times in microseconds")
    print()
    print(
        "  ",
        _pad("path", 9, False),
        _pad("n", 5, True),
        _pad("build", 12, True),
        _pad("dumps", 12, True),
        _pad("total", 12, True),
        _pad("bytes", 9, True),
        _pad("us/elem", 12, True),
    )
    print("  " + "-" * 78)

    for n in [1, 10, 100]:
        bench_legacy(n)
    print()
    for n in [1, 10, 100]:
        bench_factory(n)
    print()
    for n in [1, 10, 100]:
        bench_moved(n)
    print()
    for n in [1, 10, 100]:
        bench_parse(n)
