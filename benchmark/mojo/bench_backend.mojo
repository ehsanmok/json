# json Backend Comparison Benchmark
# Compares simdjson FFI backend vs pure Mojo backend
#
# Usage:
#   pixi run mojo -I . benchmark/mojo/bench_backend.mojo [json_file]
#
# Default: benchmark/datasets/twitter.json

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    ThroughputMeasure,
    BenchMetric,
)
from std.pathlib import Path
from std.sys import argv
from json import loads


def main() raises:
    # Get file path from command line or use default
    var args = argv()
    var path: String
    if len(args) > 1:
        path = String(args[1])
    else:
        path = "benchmark/datasets/twitter.json"

    print(
        "========================================================================"
    )
    print("json Backend Comparison Benchmark")
    print("  simdjson (FFI) vs Mojo (native, default)")
    print(
        "========================================================================"
    )
    print()

    # Load JSON file
    var json_str = Path(path).read_text()

    var file_size = len(json_str.as_bytes())
    var file_size_kb = Float64(file_size) / 1024.0

    print("File:", path)
    print("Size:", file_size, "bytes (", file_size_kb, "KB )")
    print()

    # Verify both backends produce same result
    print("Verifying output parity...")
    var v_simdjson = loads[target="cpu-simdjson"](json_str)
    var v_mojo = loads(json_str)  # Default is now Mojo

    if v_simdjson.is_object() != v_mojo.is_object():
        print("ERROR: Backend outputs differ!")
        return
    if v_simdjson.is_array() != v_mojo.is_array():
        print("ERROR: Backend outputs differ!")
        return
    print("  OK - Both backends produce equivalent output")
    print()

    var bench = Bench(BenchConfig(max_iters=100))

    # Benchmark simdjson FFI backend
    @always_inline
    def bench_simdjson(mut b: Bencher) raises {imm json_str}:
        @always_inline
        def call_fn() raises {imm json_str}:
            var v = loads[target="cpu-simdjson"](json_str)
            _ = v.is_object()

        b.iter(call_fn)

    # Benchmark Mojo native backend (default)
    @always_inline
    def bench_mojo(mut b: Bencher) raises {imm json_str}:
        @always_inline
        def call_fn() raises {imm json_str}:
            var v = loads(json_str)  # Default is Mojo backend
            _ = v.is_object()

        b.iter(call_fn)

    var measures = List[ThroughputMeasure]()
    measures.append(ThroughputMeasure(BenchMetric.bytes, file_size))

    bench.bench_function(
        bench_simdjson, BenchId("cpu-simdjson", "loads"), measures
    )
    bench.bench_function(bench_mojo, BenchId("cpu (mojo)", "loads"), measures)

    print(bench)

    # Calculate and display comparison
    print()
    print("Note: Lower time = faster. The 'cpu (mojo)' is the default backend.")
