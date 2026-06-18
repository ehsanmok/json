# Apple Metal GPU sweep benchmark.
#
# Runs `loads[target='gpu']` against a sweep of file sizes so we can
# (a) confirm chunked correctness across the 32 MB boundary and
# (b) populate the README's Apple M3 Pro row with measured numbers.
#
# Each row prints: file size in MB, chunk count, parse time in ms,
# throughput in GB/s, structural-position count.
#
# Usage:
#     pixi run -e dev mojo -I . benchmark/mojo/bench_gpu_apple.mojo \
#         benchmark/datasets/twitter.json \
#         benchmark/datasets/citm_catalog.json \
#         benchmark/datasets/twitter_large_record.json
#
# This is intentionally separate from `benchmark/mojo/bench_gpu.mojo`
# so a crashy run doesn't poison the main benchmark binary.

from std.collections import List
from std.memory import memcpy
from std.sys import argv
from std.time import perf_counter_ns

from json import loads
from json.gpu import parse_json_gpu
from json.types import JSONInput
from std.pathlib import Path


def _bench_file(path_str: String) raises:
    var path = Path(path_str)
    var raw = String(path.read_text())
    var n = len(raw.as_bytes())
    var n_mb = n // (1024 * 1024)

    print("===", path_str, "===")
    print("  size:", n_mb, "MB (", n, "bytes)")

    # Build a JSONInput once -- avoids repeating the host-side copy on
    # every iteration. Each call to parse_json_gpu still owns its own
    # input internally.
    var iters = 3 if n_mb > 64 else 5

    var best_ms = 1e18
    var best_structurals = 0
    for i in range(iters):
        # Reload bytes per iteration since parse_json_gpu takes ownership.
        var data = List[UInt8](capacity=n)
        data.resize(n, 0)
        memcpy(
            dest=data.unsafe_ptr(),
            src=raw.as_bytes().unsafe_ptr(),
            count=n,
        )
        var inp = JSONInput(data^)

        var t0 = perf_counter_ns()
        var result = parse_json_gpu(inp^, verbose=(i == 0))
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if ms < best_ms:
            best_ms = ms
            best_structurals = len(result.structural)
        print(
            "  iter", i, ":", ms, "ms,", len(result.structural), "structurals"
        )

    var gbps = (Float64(n) / 1e9) / (best_ms / 1e3)
    print(
        "  BEST:",
        best_ms,
        "ms /",
        gbps,
        "GB/s /",
        best_structurals,
        "structurals",
    )
    print()


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("Usage: bench_gpu_apple <file.json> [<file.json> ...]")
        return

    print("Apple Metal GPU sweep")
    print("======================")
    print()
    for i in range(1, len(args)):
        _bench_file(String(args[i]))
