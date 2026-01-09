# mojson CPU Benchmark
# Uses simdjson dataset: https://github.com/simdjson/simdjson/tree/master/jsonexamples
#
# Usage:
#   mojo -I . benchmark/mojo/bench_cpu.mojo [json_file]
#
# Default: benchmark/simdjson/jsonexamples/twitter.json

from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    ThroughputMeasure,
    BenchMetric,
)
from pathlib import Path
from sys import argv
from src import loads


fn main() raises:
    # Get file path from command line or use default
    var args = argv()
    var path: String
    if len(args) > 1:
        path = String(args[1])
    else:
        path = "benchmark/simdjson/jsonexamples/twitter.json"

    print(
        "========================================================================"
    )
    print("mojson CPU Benchmark")
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

    var bench = Bench(BenchConfig(max_iters=100))

    @parameter
    @always_inline
    fn bench_loads(mut b: Bencher) raises capturing:
        @parameter
        @always_inline
        fn call_fn() raises:
            var v = loads(json_str)
            _ = v.is_object()

        b.iter[call_fn]()

    var measures = List[ThroughputMeasure]()
    measures.append(ThroughputMeasure(BenchMetric.bytes, file_size))
    bench.bench_function[bench_loads](BenchId("mojson_cpu", "loads"), measures)

    print(bench)
