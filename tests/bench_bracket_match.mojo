# Benchmark: GPU vs CPU bracket matching

from time import perf_counter_ns
from gpu.host import DeviceContext
from collections import List
from memory import memcpy

from src.gpu.bracket_match import match_brackets_gpu
from src.types import JSONResult


fn generate_brackets(depth: Int, count: Int) -> List[UInt8]:
    """Generate nested brackets for benchmarking."""
    var result = List[UInt8](capacity=count * 2)

    # Pattern: alternating depths
    for i in range(count):
        # Open brackets
        for _ in range(depth):
            result.append(ord("{"))
        # Close brackets
        for _ in range(depth):
            result.append(ord("}"))

    return result^


fn cpu_bracket_match(char_types: List[UInt8]) -> List[Int32]:
    """CPU stack-based bracket matching (same as _match_brackets_fast)."""
    var n = len(char_types)
    var pair_pos = List[Int32](capacity=n)
    pair_pos.resize(n, -1)

    var stack = List[Int]()

    for i in range(n):
        var c = char_types[i]
        if c == ord("{") or c == ord("["):
            stack.append(i)
        elif c == ord("}") or c == ord("]"):
            if len(stack) > 0:
                var open_idx = stack.pop()
                pair_pos[open_idx] = Int32(i)
                pair_pos[i] = Int32(open_idx)

    return pair_pos^


fn benchmark_bracket_matching(n: Int, iterations: Int) raises:
    """Benchmark GPU vs CPU bracket matching."""
    print("=== Bracket Matching Benchmark ===")
    print("Elements:", n)
    print("Iterations:", iterations)
    print()

    # Generate test data: alternating depth 3 brackets
    var char_types = generate_brackets(depth=3, count=n // 6)
    var actual_n = len(char_types)
    print("Actual bracket count:", actual_n)
    print()

    # Warm up
    var ctx = DeviceContext()
    var d_char_types = ctx.enqueue_create_buffer[DType.uint8](actual_n)
    var h_char_types = ctx.enqueue_create_host_buffer[DType.uint8](actual_n)
    memcpy(
        dest=h_char_types.unsafe_ptr(),
        src=char_types.unsafe_ptr(),
        count=actual_n,
    )
    ctx.enqueue_copy(d_char_types, h_char_types)
    ctx.synchronize()

    # Warmup GPU
    _ = match_brackets_gpu(ctx, d_char_types.unsafe_ptr(), actual_n)

    # --- CPU Benchmark ---
    print("--- CPU (stack-based) ---")
    var cpu_times = List[Float64]()
    for _ in range(iterations):
        var t0 = perf_counter_ns()
        var result = cpu_bracket_match(char_types)
        var t1 = perf_counter_ns()
        cpu_times.append(Float64(t1 - t0) / 1e6)
        _ = len(result)

    var cpu_min = cpu_times[0]
    var cpu_sum: Float64 = 0
    for i in range(len(cpu_times)):
        var t = cpu_times[i]
        if t < cpu_min:
            cpu_min = t
        cpu_sum += t
    var cpu_avg = cpu_sum / Float64(iterations)
    print("  Min:", cpu_min, "ms")
    print("  Avg:", cpu_avg, "ms")
    print()

    # --- GPU Benchmark ---
    print("--- GPU (prefix_sum + depth) ---")
    var gpu_times = List[Float64]()
    for _ in range(iterations):
        var t0 = perf_counter_ns()
        var result = match_brackets_gpu(
            ctx, d_char_types.unsafe_ptr(), actual_n
        )
        var t1 = perf_counter_ns()
        gpu_times.append(Float64(t1 - t0) / 1e6)
        _ = len(result[1])

    var gpu_min = gpu_times[0]
    var gpu_sum: Float64 = 0
    for i in range(len(gpu_times)):
        var t = gpu_times[i]
        if t < gpu_min:
            gpu_min = t
        gpu_sum += t
    var gpu_avg = gpu_sum / Float64(iterations)
    print("  Min:", gpu_min, "ms")
    print("  Avg:", gpu_avg, "ms")
    print()

    # --- Comparison ---
    print("--- Comparison ---")
    if gpu_min < cpu_min:
        print("GPU is", cpu_min / gpu_min, "x faster")
    else:
        print("CPU is", gpu_min / cpu_min, "x faster")
    print()

    # --- Verify correctness ---
    print("--- Verification ---")
    var cpu_result = cpu_bracket_match(char_types)
    var gpu_result = match_brackets_gpu(
        ctx, d_char_types.unsafe_ptr(), actual_n
    )
    var gpu_pair_pos = gpu_result[1].copy()

    var mismatches = 0
    for i in range(actual_n):
        if cpu_result[i] != gpu_pair_pos[i]:
            if mismatches < 10:
                print(
                    "  Mismatch at",
                    i,
                    ": CPU=",
                    Int(cpu_result[i]),
                    "GPU=",
                    Int(gpu_pair_pos[i]),
                )
            mismatches += 1

    if mismatches == 0:
        print("  All pairs match!")
    else:
        print("  Total mismatches:", mismatches)
    print()
    print("=" * 40)


fn main() raises:
    # Small test
    benchmark_bracket_matching(1000, 10)
    print()

    # Medium test
    benchmark_bracket_matching(100000, 10)
    print()

    # Large test (more representative)
    benchmark_bracket_matching(1000000, 5)
