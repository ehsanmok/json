# Performance Deep Dive

This document explains why json is faster than existing parsers and the key optimizations that make it possible.

## GPU: lean pipeline across NVIDIA / AMD / Apple

On NVIDIA B200 with 804 MB `twitter_large_record.json` (same lean
pipeline runs on AMD MI355X and Apple Metal):

| Parser | Throughput | Time | Notes |
|--------|------------|------|---|
| cuJSON (CUDA C++) | 3.6 GB/s | 236 ms | baseline |
| **json GPU (pinned wall-clock)** | **~6.5 GB/s** | **~130 ms** | what `bench-gpu` reports |

*Based on warmed-up runs (3 + 100). Reproduce with `pixi run -e dev
bench-gpu benchmark/datasets/twitter_large_record.json`; pass
`--debug-timing` for the per-phase breakdown.*

## Key Optimizations

| Optimization | Impact | Description |
|--------------|--------|-------------|
| **GPU Stream Compaction** | 🔥 **Main speedup** | Reduces D2H transfer from ~160 MB to ~4 MB of position indices |
| **Pinned Memory** | H2D: ~15 ms | Uses `HostBuffer` for fast host-to-device transfer |
| **Lean pipeline** | unifies all backends | One fused kernel + positions-only stream compaction. No popcount + hierarchical prefix-sum cascade (the in-string mask is applied CPU-side by the tape adapter, byte-by-byte), no `pair_pos` array (the tape adapter does not need bracket pairs), no `char_types` companion stream out of the GPU. Same pipeline on NVIDIA, AMD, and Apple Metal. |
| **Fused Kernel** | Lower overhead | Single kernel emits both the `{}[]:,` structural bitmap and the `{}[]` open-close bitmap. |

## Why json is Faster: The Stream Compaction Advantage

### The Problem with cuJSON

cuJSON transfers **all structural character data** back to CPU:

- Input: 804MB JSON file
- Structural chars: ~58% of input = **465MB transfer**
- D2H time: **~160ms** (bottleneck)

### json's Solution

json uses **GPU stream compaction** to extract only position indices:

- Input: 804MB JSON file
- Position array: ~1 million positions × 4 bytes = **4MB transfer**
- D2H time: **minimal** (116x smaller data transfer)

This is the primary reason for the 2x overall speedup.

## Detailed Timing Breakdown

### cuJSON Pipeline (~236ms total)

```
cuJSON breakdown (average):
├─ H2D transfer:       ~15 ms   (804MB → GPU)
├─ Validation:          ~2 ms   (GPU)
├─ Tokenization:        ~6 ms   (GPU)
├─ Parser:              ~2 ms   (GPU)
└─ D2H transfer:      ~160 ms   (465MB → CPU, bottleneck)
────────────────────────────────
TOTAL:                ~236 ms
Throughput:           3.6 GB/s
```

### json GPU Pipeline (~130 ms total, lean)

```
json pinned breakdown (B200):
├─ H2D + alloc:       ~15 ms   (804 MB → GPU, pinned memory + buffer alloc)
├─ Fused kernel:       ~2 ms   ({}[]:,/{}[]  bitmaps in one launch)
├─ Stream compact:    ~95 ms   (popcount + block prefix sum + scatter, positions-only)
└─ D2H + finalize:    ~17 ms   (~4 MB positions → CPU + JSONResult fill)
────────────────────────────────
TOTAL:               ~130 ms
Throughput:           ~6.5 GB/s (pinned wall-clock)
```

There is no GPU-side in-string mask and no CPU bracket-matching
pass. The tape adapter walks the byte stream once to apply the
escape state machine and stage 2 directly consumes
`gpu_result.structural`; `pair_pos` is left as a placeholder for
legacy iterator ABI compatibility.

## Architecture Comparison

| Aspect | cuJSON | json |
|--------|--------|--------|
| **Input memory** | Pinned (cudaMallocHost) | Pinned (HostBuffer) |
| **H2D transfer** | ✓ (15 ms) | ✓ (15 ms) |
| **GPU kernels** | Validation + Tokenization | Single fused kernel emits `{}[]:,` + `{}[]` bitmaps |
| **Position extraction** | ❌ (transfers all data) | ✅ **GPU stream compaction (positions only)** |
| **D2H transfer** | 465 MB (~160 ms) | 4 MB (~15 ms) |
| **Bracket / pair-match** | GPU (Parser kernel) | not done -- the tape adapter walks the byte stream once on the CPU and does not need a `pair_pos` array |

## Performance Metrics Explained

`pixi run bench-gpu` reports a single `std.benchmark.Bench` table with
four rows so you can see where time goes across the pipeline:

| Row | What It Includes | Use Case |
|-----|------------------|----------|
| **from host bytes: memcpy + parse (wall-clock)** | host→pinned memcpy + `parse_json_gpu_from_pinned` | Realistic "bytes in memory → parsed" cost |
| **parse_json_gpu_from_pinned (pinned, wall-clock)** | H2D + GPU kernels + stream compaction + D2H + CPU bracket matching | Apples-to-apples comparison with cuJSON (both assume pinned input) |
| **parse_json_gpu_from_pinned (device-only)** | Same call, timed via `DeviceContext.execution_time` (CUDA events) | Pure device-queue time, excludes host-side CPU post-processing |
| **loads[target='gpu'] (from json.gpu)** | Everything + `Value` tree construction on CPU | Real-world application performance |

### Why Four Rows?

1. **Pinned wall-clock (~121 ms, 7.0 GB/s):** apples-to-apples with cuJSON
   (both assume pinned input). This is the headline GPU-parse number.
2. **Pinned device-only (~100 ms, ~8 GB/s):** drops the host-side
   bracket-matching and list-build work. Use this to compare against
   kernel-only timings from other frameworks.
3. **from host bytes (~280 ms, ~2.9 GB/s):** adds the realistic
   host→pinned memcpy (~120 ms for 804 MB on DDR5).
4. **Full `loads[target='gpu'] (from json.gpu)` (~900 ms, ~1.0 GB/s):** adds the
   CPU-bound `Value` tree construction on top of everything.

Pass `--debug-timing` to get a per-phase breakdown (H2D, GPU kernels,
position extraction, bracket matching, total) printed alongside the
summary table:

```bash
pixi run bench-gpu -- --debug-timing benchmark/datasets/twitter_large_record.json
```

## Benchmark Results

### GPU Performance

GPU benchmarks are only meaningful for large files (>100 MB). For
smaller files, GPU launch overhead dominates and the results are
not representative.

804 MB `twitter_large_record.json`:

| Platform | Throughput | Pipeline |
|---|---:|---|
| AMD MI355X | 13 GB/s | lean (single-shot) |
| NVIDIA B200 | ~6.5 GB/s | lean (single-shot, pinned wall-clock) |
| Apple M3 Pro | 3.1 GB/s | lean Metal (chunked at 64 MB) |

All backends run the same lean pipeline: one fused kernel + a
positions-only stream-compaction extract. The GPU does not build an
in-string mask and does not produce a `char_types` companion stream
(which would have been a ~66 MB D2H copy on `twitter_large_record`).
`gpu/tape_adapter.mojo` walks the byte stream once on the CPU to
apply the escape state machine and feeds stage 2 of the CPU pipeline
to construct the tape -- the same stage 2 the CPU-only path uses.

## Typed serde

`serialize_json` and `deserialize_json` go straight between a struct
and JSON bytes. Reading in particular used to go the long way round --
parse into a tape-backed `Document`, wrap it in `Value`, walk that --
so decoding built a whole document representation that was discarded
one field later, took an atomic refcount touch per field access, and
allocated a `String` for every object key merely to compare it against
a field name.

`json/reader.mojo` is a forward byte cursor that writes each value into
its final address. `loads` is unchanged and still produces a
`Document`, because navigating a document and decoding one into known
types are different jobs.

### Numbers

Five record shapes, 100 records each, median of seven calibrated
batches, Apple M3 Pro, `-D ASSERT=none`. Reproduce with
`pixi run -e dev bench-serde`.

| Shape | Bytes | `deserialize_json` | `loads` + `Value` walk | `serialize_json` |
|---|---:|---:|---:|---:|
| message | 16 KB | 15.6 us | 53.2 us | 12.4 us |
| document | 47 KB | 60.8 us | 186.0 us | 31.5 us |
| telemetry | 43 KB | 82.0 us | 141.6 us | 117.4 us |
| strings | 43 KB | 61.2 us | 109.6 us | 33.7 us |
| event | 27 KB | 38.8 us | 93.5 us | 14.7 us |

### What the read path costs, and what it stopped costing

Before this work the cost was about 34 ns per token on every shape,
regardless of what the tokens were. That flatness was the tell: it was
fixed per-token overhead, not work proportional to the data. It is now
about 15 ns per token on the document shape. In order of effect:

1. **A list is sized once, not grown from nothing.** `List` doubles
   from capacity zero, so an eight-element array cost four allocations
   and three copies; filling one that way measured 195 ns against 51 ns
   with the capacity right to begin with. JSON does not announce an
   array's length, so eight is reserved on the first element and
   `append` doubles from there.
2. **Short integers skip the full number scanner.** Reaching a value
   through the scanner cost four outlined calls, a vector peek for an
   eight-digit block a short integer never has, and a forty-eight byte
   token to unpack. `read_int` now commits when it has itself
   established the whole token -- optional minus, leading digit in
   1-9, at most eighteen digits, a terminator that cannot continue a
   number -- and defers everything else to the scanner, which stays
   the only authority on the grammar and its messages.
3. **A string is scanned once.** Finding the closing quote and
   classifying the body were separate passes, and the second only
   vectorizes at sixteen bytes, which real keys and values are not.
   They are fused; the flags are masked to the bytes before the quote,
   without which a control character in the *next* token would be
   attributed to this string.
4. **Each byte between members is looked at once.** `next_member` ran
   the whitespace scan three times per member from positions where
   nothing had been consumed.
5. **A number's digits are accumulated while being validated**, rather
   than walked once for the grammar and again for the value.
6. **Field names are matched at compile time**, with the reader passed
   immutably so the unrolled match loop does not reload its state after
   every candidate.

### What the float writer cost, and what it costs now

Writing floats is still the slowest thing this library does, but by a
smaller margin than it was. The telemetry shape, 32 floats per record,
serialized in 166 us where the strings shape of the same size in bytes
took 34. It now takes 117.

Two things were wrong, and the obvious one was not the cost.

Indexing a `comptime` array copies the whole array into the caller's
frame before the first read, which for the Grisu2 cached powers is
about a kilobyte per float. Moving both tables into constant data made
digit generation *slower*, not faster, and was reverted. The integer
writer did gain from the same change, which is what took the document
shape from 37 us to 31 us.

The real cost was dividing by a value the compiler could not see.
Grisu2's integral loop divided by `pow10[kappa - 1]`, so every digit
was a real 32-bit division. Unrolling the loop over the ten possible
digit counts makes each divisor a literal and each division the
multiply-and-shift a compiler emits for one. With a cheaper digit
count and one fewer table, `shortest_digits` went from 51.2 to 42.1 ns
per float on random mantissas, and rather more than that on the short
decimals real payloads are made of.

## CPU Performance

json has one CPU code path served by `loads(target='cpu')`: a pure
Mojo two-pass parser that emits a packed `Document` tape. Stage 1
finds structural positions; stage 2 walks them and writes typed
tape entries. SIMD stage 1 is the default; opt into the scalar
oracle with `parse_cpu_native_tape[force_scalar=True]` (it exists
mainly to validate the SIMD path under fuzzing).

`Value` is a tape-backed view over the resulting `Document`,
sharing it via `ArcPointer`. Children are computed by walking
`Document.tape` -- there is no on-access re-parse, no raw
substring rescan, and no duplicate-key collapse. Strings are stored
as `(offset, length)` slices into the original input and only
materialise an owned `String` when the bytes need unescaping or the
caller asks for one.

The GPU pipeline emits the same `Document` shape, so CPU and GPU
agree on one DOM representation, and a `Value` from either backend is
just a stable index into the same tape. Mutating one converts it to
the owned tree once, not once per write. A nested write is addressed
by pointer -- `doc.set_at("/a/b", v)` -- because a subscript hands
back an independent value rather than a handle into its parent.

### Benchmark methodology

`pixi run -e dev bench-cpu <file>` reports the same `simdjson` C++
parser as the reference, then the Mojo parser, both under the same
protocol:

* **3 warmup + 100 measured iterations** per workload.
* Throughput reported as **min-time-derived GB/s** (matches the
  upstream simdjson convention).
* Mojo's parser consumes its input by value, so the bench loop
  pre-builds a `List[String]` of independent copies outside the
  timed region. The simdjson side reuses one buffer because its
  parser does not consume the input.
* Mojo bench binaries are built with `mojo build -D ASSERT=none`,
  which strips the Mojo stdlib's safety asserts (the default
  `ASSERT=safe` keeps them in for development). This matches
  simdjson C++'s `-O3` posture; running with the default
  `ASSERT=safe` build is roughly 20-37% slower on these workloads
  and is not an apples-to-apples comparison. `pixi run -e dev
  bench-cpu` already wires the flag in.

Two workloads:

* **`parse_only`** -- `loads(...)` and peek the root tag. Measures
  parse cost in isolation. The compiler can't elide the parse
  because the root tag is touched.
* **`parse_traverse`** -- parse and recursively visit every leaf
  via the public API (`array_items` / `object_items` / `*_value`).
  This is what real consumers do, and on the tape representation
  it adds only a small constant on top of `parse_only`.

### Numbers (x86 host, Mojo native two-pass)

| Corpus | Size | `parse_only` (simd) | `parse_traverse` (simd) |
|---|---|---|---|
| `twitter.json` | 616 KB | 0.598 ms / 1.06 GB/s | 0.985 ms / 0.64 GB/s |
| `citm_catalog.json` | 1.7 MB | 1.109 ms / 1.56 GB/s | 1.980 ms / 0.87 GB/s |

* The Mojo `parse_traverse` cost is ~2x the `parse_only` cost --
  traversal walks every tape slot and materializes zero-copy keys
  / strings on demand. There is no on-access re-parse, no raw
  substring rescan, and no per-iteration allocation other than the
  document itself.
* simdjson's `target='cpu-simdjson'` FFI shim is intentionally not
  in this table; the FFI marshalling cost dominates and it has not
  been competitive with the native Mojo path for several releases.
  The simdjson C++ reference baseline is reproducible via
  `pixi run -e dev bench-cpu <file>`, which runs the simdjson C++
  binary first and then the Mojo parser under the same protocol.

### Key CPU optimizations

Stage 1 (structural indexing) -- now ~5+ GB/s in isolation:

1. **PSHUFB-style nibble classifier** -- a single
   `SIMD._dynamic_shuffle` pair plus an OR replaces the per-byte
   `eq` chain that classified `{ } [ ] : , " \\` and whitespace.
2. **Branchless 64-byte chunks** -- with `prefix_xor64` and
   `find_escape_mask64` (the simdjson-style escape / in-string
   branchless propagation), we process 64 bytes per iteration with
   no per-marker scalar dispatch in the common case.
3. **Two fast paths** -- chunks with no markers and no active
   string state skip all bit-twiddling; chunks with no quotes /
   backslashes and no active string emit structurals directly
   without escape computation.

Stage 2 (tape emission):

4. **Zero-copy clean strings and keys** -- `TAPE_TAG_STRING` and
   `TAPE_TAG_KEY_INLINE` store `(offset, length)` slices into
   `Document.input`. No allocation, no `memcpy`, no
   `string_pool` / `key_pool` append on the hot path. Only
   strings / keys that actually contain escapes spill to the
   side pools.
5. **SWAR 8-digit integer parser** -- 8 ASCII digit bytes loaded
   as a `SIMD[UInt8, 8]`, multiplied by a power-of-10 vector,
   reduced with a single `reduce_add`. Compiles to a tight NEON
   `umlal+addv` sequence; AVX2 gets `vpmulld+horizontal sum`.
6. **SIMD backslash scan** -- the scalar "does this string contain
   `\\`?" loop is replaced with a 32-byte `eq+reduce_or` scan,
   exiting on the first chunk in the no-escape case.
7. **SIMD whitespace skip with scalar prelude** -- 4-byte scalar
   prelude for the dense-JSON case (0-3 ws bytes), then a 32-byte
   SIMD body using `pack_bits` + `count_trailing_zeros` for
   pretty-printed whitespace runs.
8. **Bulk `memcpy` flush** -- when a container closes, its
   children's headers are copied from `headers_scratch` into
   `doc.tape` in one `memcpy` instead of one append per header.

### What's left to close the gap

The remaining gap on `parse_only` is algorithmic:

1. **No Eisel-Lemire float fast path.** `_emit_number` still spills
   to `atof` for floats. Eisel-Lemire would parse most floats
   without allocation; integer parsing already uses SWAR.
2. **Stage 2 still re-validates byte content** between adjacent
   structurals to reject inputs like `[1foo, 2]`. simdjson trusts
   stage 1's structural index fully; doing the same here would
   remove the per-value `_skip_ws` re-scan but requires stage 1 to
   detect non-whitespace, non-structural bytes outside strings.
3. **AVX-512 64-byte chunks** are not yet enabled on hosts that
   support `vpternlogq`; the 32-byte NEON / AVX2 path is what we
   measure today.

## When to Use GPU vs CPU

| File Size | Recommended Backend | Reason |
|-----------|---------------------|--------|
| < 1 MB | **CPU (simdjson)** | GPU launch overhead dominates |
| 1-100 MB | **CPU or GPU** | Comparable performance |
| > 100 MB | **GPU** | 2x faster than cuJSON, 3-5x faster than CPU |

## Optimization Techniques

### 1. GPU Stream Compaction

**Problem:** After identifying structural characters on GPU, we need their positions on CPU for bracket matching.

**Naive approach:** Transfer entire structural character bitmap (58% of input size)

**Optimized approach:**
1. Create position bitmap on GPU
2. Use parallel prefix sum to compute output positions
3. Compact positions into dense array on GPU
4. Transfer only compact position array to CPU

**Result:** 116x reduction in D2H transfer size (465MB → 4MB)

### 2. Pinned Memory

Using `HostBuffer` (pinned memory) for H2D transfers:

- Pinned: ~15ms for 804MB
- Pageable: ~110ms for 804MB
- **Speedup:** 7.3x faster

### 3. Block-level Prefix Sums (stream compaction)

The positions-only stream compaction needs an exclusive prefix sum
over per-word popcounts so each thread knows where to scatter its
positions:

1. Each block computes a local prefix sum using `block.prefix_sum`.
2. Block totals are propagated up the hierarchy and added back into
   per-block prefixes.
3. Single-pass algorithm, minimal synchronization.

### 4. Fused Kernel

A single kernel walks 32 input bytes per thread and emits two
32-bit-per-thread bitmaps:

- The raw `{}[]:,` structural bitmap consumed by stream compaction.
- The `{}[]` open-close bitmap (reserved for downstream use).

In-string detection lives on the CPU side
(`gpu/tape_adapter._result_to_index`), where the same byte walk that
stage 2 already needs runs the correct escape state machine.

### 5. Minimize Memory Allocations

- Pre-allocate GPU buffers based on input size
- Reuse `DeviceContext` across operations
- Use `String(unsafe_from_utf8=bytes^)` for bulk string construction

### 6. Hybrid GPU/CPU Pipeline

- **GPU:** parallel bitmap operations + stream compaction.
- **CPU:** the escape state machine + tape construction in the
  existing stage 2 pass.
- **Key insight:** keep the GPU on the work it parallelises well
  (structural classification + scatter), and let the CPU handle
  byte-by-byte state that doesn't parallelise cleanly inside a
  single 32-byte chunk.

## Performance Variance

GPU performance can vary between runs due to:

- **Cold-start overhead:** First GPU run ~200ms slower (GPU initialization)
- **Thermal throttling:** GPU frequency varies with temperature
- **Scheduling:** CUDA stream scheduling can introduce variance

**Solution:** Always measure with warm-up runs and report averages.

## Future Optimizations

Potential improvements for even better performance:

1. **Apple UMA-aware single-buffer pipeline:** drop the H2D leg on
   unified-memory hosts by writing directly into a unified-memory
   `DeviceBuffer`.
2. **Multi-GPU / multi-stream overlap:** launch the next chunk's
   H2D while the previous chunk's kernel runs.
3. **Streaming parser:** Process chunks as they arrive.
4. **Zero-copy Value tree:** Build tree directly on GPU memory.

## Benchmark Reproducibility

All benchmarks are reproducible using pinned git submodules:

```bash
# Clone the repo
git clone https://github.com/ehsanmok/json.git && cd json

# Clone cuJSON (optional, for the head-to-head benchmark)
cd benchmark && git clone https://github.com/AutomataLab/cuJSON.git && cd ..

# Build comparison benchmark (lives in the dev feature)
pixi run -e dev build-cujson

# Run benchmarks
pixi run bench-gpu-cujson benchmark/datasets/twitter_large_record.json
```

See [benchmark/readme.md](../benchmark/readme.md) for complete setup instructions.

## Hardware Requirements

- **GPU:** NVIDIA (CUDA 7.0+, tested on B200, H100, A100), AMD (ROCm 6+, tested on MI355X), or Apple Silicon (Metal, tested on M3 Pro).
- **CUDA:** Latest CUDA toolkit (for NVIDIA).
- **Apple Metal:** Xcode with Metal Toolchain registered. On Xcode 26.x, run `xcodebuild -downloadComponent MetalToolchain` once if `mojo build` reports `Metal Compiler failed to compile metallib`.
- **Memory:** At least 2x your largest JSON file size (for GPU buffers).

## References

- [simdjson](https://github.com/simdjson/simdjson) - CPU JSON parser
- [cuJSON](https://github.com/AutomataLab/cuJSON) - GPU JSON parser (baseline comparison)
- [GPU stream compaction](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back) - Decoupled look-back algorithm
