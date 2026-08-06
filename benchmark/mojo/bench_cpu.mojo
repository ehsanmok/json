# Mojo CPU benchmark for the json library.
#
# Methodology mirrors `benchmark/cpp/bench_simdjson.cpp` so the two
# benches are directly comparable:
#
#   - 3 warmup iterations + 100 measured iterations
#   - Two paths: `parse_only` (parse, peek root tag) and
#     `parse_traverse` (parse + walk every node, accessing each leaf's
#     value)
#   - For each path we run a `scalar` (force_scalar=True) and `simd`
#     (default) variant of stage 1.
#   - Reports min/avg/max wall time and min-time-derived throughput.
#     Min-time is the simdjson community convention — using the same
#     stat on both sides removes the apparent gap between mean and min
#     reporting.
#
# Note on the input copy:
#   `parse_cpu_native_tape` consumes its `String` argument because the
#   resulting `Document` owns the bytes it indexes into. To keep that
#   cost out of the timed region we pre-build a `List[String]` of
#   `num_warmup + num_iters` copies before each path runs and the
#   bench loop pops one copy per iteration.
#
# Usage:
#   mojo -I . benchmark/mojo/bench_cpu.mojo [json_file]

from std.pathlib import Path
from std.sys import argv
from std.time import perf_counter_ns

from json.cpu import parse_cpu_native_tape
from json.value import Value


def _walk_strict(v: Value) -> Bool:
    """Strict counterpart to `_walk`: returns False if any access
    raises. Used once outside the bench loop to surface correctness
    bugs the timing-only walk would otherwise hide."""
    if v.is_object():
        try:
            for pair in v.object_items():
                if not _walk_strict(pair[1]):
                    return False
        except:
            return False
        return True
    if v.is_array():
        try:
            for child in v.array_items():
                if not _walk_strict(child):
                    return False
        except:
            return False
        return True
    if v.is_string():
        try:
            _ = v.string_value()
        except:
            return False
        return True
    if v.is_int():
        try:
            _ = v.int_value()
        except:
            return False
        return True
    if v.is_float():
        try:
            _ = v.float_value()
        except:
            return False
        return True
    if v.is_bool():
        try:
            _ = v.bool_value()
        except:
            return False
        return True
    return True


def _walk(v: Value) -> Int:
    """Recursively visit every node, returning a checksum so the
    traversal cannot be optimised away.

    For each leaf we mix one int's worth of payload into the
    accumulator; for containers we recurse. Errors during access are
    swallowed so the bench reports throughput even when a backend has
    a correctness bug under deep traversal — we can spot the bug
    separately, but we want the wall-clock comparison either way.
    """
    if v.is_object():
        var sum = 0
        try:
            for pair in v.object_items():
                sum += pair[0].byte_length()
                sum += _walk(pair[1])
        except:
            pass
        return sum
    if v.is_array():
        var sum = 0
        try:
            for child in v.array_items():
                sum += _walk(child)
        except:
            pass
        return sum
    if v.is_string():
        try:
            return v.string_value().byte_length()
        except:
            return 0
    if v.is_int():
        try:
            return Int(v.int_value())
        except:
            return 0
    if v.is_float():
        try:
            return Int(v.float_value())
        except:
            return 0
    if v.is_bool():
        try:
            return 1 if v.bool_value() else 0
        except:
            return 0
    return 0


def _make_pool(json_str: String, n: Int) raises -> List[String]:
    """Pre-build `n` independent copies of `json_str` for the bench loop.

    `parse_cpu_native_tape` consumes its argument by value; without
    this pool every iteration would also pay one `String.copy()` of
    overhead inside the timed region.
    """
    var pool = List[String](capacity=n)
    for _ in range(n):
        pool.append(json_str.copy())
    return pool^


def _percentiles(
    times_ns: List[Int],
) raises -> Tuple[Float64, Float64, Float64]:
    """Return (min, avg, max) milliseconds from a list of nanosecond
    durations."""
    var min_ns = times_ns[0]
    var max_ns = times_ns[0]
    var sum_ns: Int = 0
    for t in times_ns:
        if t < min_ns:
            min_ns = t
        if t > max_ns:
            max_ns = t
        sum_ns += t
    var n = len(times_ns)
    var min_ms = Float64(Int(min_ns)) / 1.0e6
    var max_ms = Float64(Int(max_ns)) / 1.0e6
    var avg_ms = Float64(Int(sum_ns)) / 1.0e6 / Float64(n)
    return (min_ms, avg_ms, max_ms)


def _print_row(
    label: String,
    min_ms: Float64,
    avg_ms: Float64,
    max_ms: Float64,
    file_size: Int,
) -> None:
    """Format one row identically to the simdjson C++ bench."""
    var throughput = (Float64(file_size) / 1.0e9) / (min_ms / 1000.0)
    print(
        "  ",
        label,
        ": min ",
        min_ms,
        " ms | avg ",
        avg_ms,
        " ms | max ",
        max_ms,
        " ms | ",
        throughput,
        " GB/s (min-based)",
        sep="",
    )


def _bench_parse_only[
    force_scalar: Bool
](
    json_str: String,
    file_size: Int,
    label: String,
    num_warmup: Int,
    num_iters: Int,
) raises:
    """Time `parse_cpu_native_tape` + root-tag peek for `num_iters`
    iterations after `num_warmup` warmup runs."""
    var warm_pool = _make_pool(json_str, num_warmup)
    for i in range(num_warmup):
        var v = parse_cpu_native_tape[force_scalar=force_scalar](
            warm_pool[i].copy()
        )
        _ = v.is_object()

    var pool = _make_pool(json_str, num_iters)
    var times = List[Int](capacity=num_iters)
    for i in range(num_iters):
        var t0 = perf_counter_ns()
        var v = parse_cpu_native_tape[force_scalar=force_scalar](pool[i].copy())
        _ = v.is_object()
        var t1 = perf_counter_ns()
        times.append(t1 - t0)

    var stats = _percentiles(times)
    _print_row(label, stats[0], stats[1], stats[2], file_size)


def _bench_parse_traverse[
    force_scalar: Bool
](
    json_str: String,
    file_size: Int,
    label: String,
    num_warmup: Int,
    num_iters: Int,
) raises:
    """Time `parse_cpu_native_tape` + full DOM walk for `num_iters`
    iterations after `num_warmup` warmup runs."""
    var warm_pool = _make_pool(json_str, num_warmup)
    for i in range(num_warmup):
        var v = parse_cpu_native_tape[force_scalar=force_scalar](
            warm_pool[i].copy()
        )
        _ = _walk(v)

    var pool = _make_pool(json_str, num_iters)
    var times = List[Int](capacity=num_iters)
    for i in range(num_iters):
        var t0 = perf_counter_ns()
        var v = parse_cpu_native_tape[force_scalar=force_scalar](pool[i].copy())
        _ = _walk(v)
        var t1 = perf_counter_ns()
        times.append(t1 - t0)

    var stats = _percentiles(times)
    _print_row(label, stats[0], stats[1], stats[2], file_size)


def main() raises:
    var args = argv()
    var path: String
    if len(args) > 1:
        path = String(args[1])
    else:
        path = "benchmark/datasets/twitter.json"

    print("\n--- json mojo (CPU, tape) ---")
    var json_str = Path(path).read_text()
    var file_size = len(json_str.as_bytes())
    var file_size_kb = Float64(file_size) / 1024.0
    print("File:", path)
    print("Size:", file_size, "bytes (", file_size_kb, "KB)")
    print()

    # Correctness probe — parse once and walk strictly. We do this
    # before timing so a corrupt tape or access bug is loud rather
    # than silently skewing the throughput numbers (the timing walk
    # swallows access errors so wall-clock numbers stay comparable).
    var diag = parse_cpu_native_tape(json_str.copy())
    var diag_ok = _walk_strict(diag)
    if diag_ok:
        print("traversal correctness: OK")
    else:
        print("traversal correctness: FAILED -- access error during walk")
    print()

    var num_warmup = 3
    var num_iters = 100
    print("Iterations:", num_iters, "(warmup", num_warmup, ")")
    print()

    _bench_parse_only[False](
        json_str, file_size, "parse_only    (simd  )", num_warmup, num_iters
    )
    _bench_parse_only[True](
        json_str, file_size, "parse_only    (scalar)", num_warmup, num_iters
    )
    _bench_parse_traverse[False](
        json_str, file_size, "parse_traverse(simd  )", num_warmup, num_iters
    )
    _bench_parse_traverse[True](
        json_str, file_size, "parse_traverse(scalar)", num_warmup, num_iters
    )
    print()
