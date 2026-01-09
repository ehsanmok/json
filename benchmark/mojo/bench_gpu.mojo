# mojson GPU Benchmark
# Uses cuJSON dataset: https://github.com/AutomataLab/cuJSON
#
# Usage:
#   mojo -I . benchmark/mojo/bench_gpu.mojo [json_file]

from pathlib import Path
from sys import argv
from time import perf_counter_ns
from memory import memcpy
from collections import List

from src import loads
from src.gpu import parse_json_gpu, parse_json_gpu_from_pinned
from src.types import JSONInput
from gpu.host import DeviceContext


fn main() raises:
    var args = argv()
    var path: String
    if len(args) > 1:
        path = String(args[1])
    else:
        path = "benchmark/datasets/twitter.json"

    print()
    print("--- mojson GPU (Mojo) ---")

    # Load JSON file
    var content = Path(path).read_text()
    var size = len(content)
    var size_mb = Float64(size) / 1024.0 / 1024.0

    print("File:", path)
    print("Size:", size, "bytes (", size_mb, "MB )")
    print()

    var num_iters = 10
    if size_mb > 100:
        num_iters = 5
    elif size_mb > 10:
        num_iters = 10

    # Warmup
    print("Warming up GPU...")
    for _ in range(2):
        var result = loads[target="gpu"](content)
        _ = result.is_object()
    print()

    # ===== Detailed timing of raw GPU parser =====
    print("=== Raw GPU Parser Timing ===")
    var data = content.as_bytes()
    var n = len(data)

    # Pre-allocate bytes for raw parser timing
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

    # ===== Pinned memory path (skip memcpy) =====
    print("=== Pinned Memory Path (Skip memcpy) ===")
    var ctx = DeviceContext()
    var pinned_min_time: UInt = 0xFFFFFFFFFFFFFFFF
    for _ in range(3):
        # Load directly into pinned memory
        var h_input = ctx.enqueue_create_host_buffer[DType.uint8](n)
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

    # ===== Full loads benchmark =====
    print("=== Full loads[target='gpu'] Benchmark ===")
    print("Running", num_iters, "iterations...")
    print()

    var total_time: UInt = 0
    var min_time: UInt = 0xFFFFFFFFFFFFFFFF
    var max_time: UInt = 0

    for i in range(num_iters):
        var start = perf_counter_ns()
        var result = loads[target="gpu"](content)
        var end = perf_counter_ns()

        _ = result.is_object()
        var elapsed = end - start
        total_time += elapsed
        if elapsed < min_time:
            min_time = elapsed
        if elapsed > max_time:
            max_time = elapsed

    var avg_ms = Float64(total_time) / Float64(num_iters) / 1_000_000.0
    var min_ms = Float64(min_time) / 1_000_000.0
    var max_ms = Float64(max_time) / 1_000_000.0
    var throughput = Float64(size) / Float64(min_time) * 1e9 / 1e9

    print("TOTAL (ms):            ", min_ms)
    print("Throughput (GB/s):     ", throughput)
    print()
    print("Stats:")
    print("  Min:", min_ms, "ms")
    print("  Avg:", avg_ms, "ms")
    print("  Max:", max_ms, "ms")
    print()
