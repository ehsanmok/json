# json GPU Benchmark
# Uses cuJSON dataset: https://github.com/AutomataLab/cuJSON
#
# Usage:
#   mojo -I . benchmark/mojo/bench_gpu.mojo [json_file]

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    ThroughputMeasure,
    BenchMetric,
    Unit,
)
from std.pathlib import Path
from std.sys import argv
from std.time import perf_counter_ns
from std.memory import memcpy
from std.collections import List

from json import loads
from json.gpu import parse_json_gpu, parse_json_gpu_from_pinned
from json.types import JSONInput
from std.gpu.host import DeviceContext


def main() raises:
    var args = argv()
    var path: String
    if len(args) > 1:
        path = String(args[1])
    else:
        path = "benchmark/datasets/twitter.json"

    print()
    print("=" * 72)
    print("json GPU Benchmark")
    print("=" * 72)
    print()

    # Load JSON file
    var content = Path(path).read_text()
    var size = content.byte_length()
    var size_mb = Float64(size) / 1024.0 / 1024.0

    print("File:", path)
    print("Size:", size, "bytes (", size_mb, "MB )")
    print()

    # Warmup GPU
    print("Warming up GPU...")
    for _ in range(2):
        var result = loads[target="gpu"](content)
        _ = result.is_object()
    print()

    # ===== Raw GPU Parser Timing (manual for precision) =====
    print("=== Raw GPU Parser Timing ===")
    var data = content.as_bytes()
    var n = len(data)

    var raw_min_time: UInt = 0xFFFFFFFFFFFFFFFF
    for _ in range(3):
        var bytes = List[UInt8](capacity=n)
        bytes.resize(n, 0)
        memcpy(dest=bytes.unsafe_ptr(), src=data.unsafe_ptr(), count=n)
        var input_obj = JSONInput(bytes^)

        var start = perf_counter_ns()
        var result = parse_json_gpu(input_obj^)
        var end = perf_counter_ns()

        var elapsed = end - start
        if elapsed < raw_min_time:
            raw_min_time = elapsed

        _ = len(result.structural)

    var raw_ms = Float64(raw_min_time) / 1_000_000.0
    var raw_throughput = Float64(size) / Float64(raw_min_time) * 1e9 / 1e9
    print("Raw GPU parse time (ms):", raw_ms)
    print("Raw GPU throughput (GB/s):", raw_throughput)
    print()

    # ===== Pinned Memory Path (manual for precision) =====
    print("=== Pinned Memory Path (Skip memcpy) ===")
    var ctx = DeviceContext()
    # Allocate pinned buffer ONCE outside the loop (pinned alloc is slow)
    var h_input = ctx.enqueue_create_host_buffer[DType.uint8](n)
    var pinned_min_time: UInt = 0xFFFFFFFFFFFFFFFF
    for _ in range(3):
        # Only memcpy each iteration, reuse the pinned buffer
        memcpy(dest=h_input.unsafe_ptr(), src=data.unsafe_ptr(), count=n)

        var start = perf_counter_ns()
        var result = parse_json_gpu_from_pinned(ctx, h_input, n)
        var end = perf_counter_ns()

        var elapsed = end - start
        if elapsed < pinned_min_time:
            pinned_min_time = elapsed

        _ = len(result.structural)

    var pinned_ms = Float64(pinned_min_time) / 1_000_000.0
    var pinned_throughput = Float64(size) / Float64(pinned_min_time) * 1e9 / 1e9
    print("Pinned memory parse time (ms):", pinned_ms)
    print("Pinned memory throughput (GB/s):", pinned_throughput)
    print("Speedup vs raw:", raw_ms / pinned_ms, "x")
    print()

    # ===== Full loads[target='gpu'] Benchmark using official API =====
    print("=== Full loads[target='gpu'] Benchmark ===")
    print()

    # Configure based on file size
    var max_iters = 100
    if size_mb > 100:
        max_iters = 10
    elif size_mb > 10:
        max_iters = 20

    var bench = Bench(BenchConfig(max_iters=max_iters))

    @parameter
    @always_inline
    def bench_gpu_loads(mut b: Bencher) raises capturing:
        @parameter
        @always_inline
        def call_fn() raises:
            var v = loads[target="gpu"](content)
            _ = v.is_object()

        b.iter[call_fn]()

    var measures = List[ThroughputMeasure]()
    measures.append(ThroughputMeasure(BenchMetric.bytes, size))
    bench.bench_function[bench_gpu_loads](
        BenchId("json_gpu", "loads[target='gpu']"), measures
    )

    print(bench)
