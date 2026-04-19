# json Benchmarks

Comprehensive benchmarks comparing json against reference implementations (cuJSON for GPU, simdjson for CPU).

## Quick Start

```bash
# GPU benchmark (json vs cuJSON) - apples-to-apples comparison
pixi run bench-gpu-cujson benchmark/datasets/twitter_large_record.json

# GPU benchmark (json only, 4-row Bench report)
pixi run bench-gpu benchmark/datasets/twitter_large_record.json

# GPU benchmark with per-phase timing breakdown
pixi run bench-gpu -- --debug-timing benchmark/datasets/twitter_large_record.json

# CPU benchmark (json vs simdjson)
pixi run bench-cpu benchmark/datasets/twitter.json
```

## Setup

### 1. Clone the Repo

```bash
git clone https://github.com/ehsanmok/json.git
cd json
```

### 2. Install Dependencies

```bash
pixi install  # Installs Mojo + libsimdjson (from conda-forge), builds FFI wrapper
```

### 3. Build cuJSON (for GPU comparison benchmarks)

cuJSON is not bundled. Clone it manually into `benchmark/cuJSON`, then
build via the dev feature (which provides `git` and the nvcc wrapper
tasks; nvcc itself must be on PATH from your CUDA install):

```bash
cd benchmark && git clone https://github.com/AutomataLab/cuJSON.git
cd .. && pixi run -e dev build-cujson
```

This builds `benchmark/cuJSON/build/cujson_benchmark`. Tested against
cuJSON commit `2ac7d3dcd7ad1ff64ebdb14022bf94c59b3b4953`.

## Benchmark Results

### GPU: json vs cuJSON (NVIDIA B200)

**Dataset:** 804MB `twitter_large_record.json`

| Parser | Time | Throughput | Speedup |
|--------|------|------------|---------|
| cuJSON (CUDA C++) | 182 ms | 4.6 GB/s | baseline |
| **json GPU** | **103 ms** | **8.2 GB/s** | **1.8x** |

### CPU: json (simdjson FFI)

**Dataset:** 804MB `twitter_large_record.json`

| Platform | Time | Throughput |
|----------|------|------------|
| Apple M3 Pro | 530 ms | 1.5 GB/s |
| Intel Xeon 6972P | 2747 ms | 0.3 GB/s |

## Important: GPU Benchmarks Require Large Files

**GPU benchmarks are only meaningful for files >100MB.** For smaller files, GPU launch overhead dominates and results are misleading. Always use large datasets (e.g., `twitter_large_record.json`) for GPU performance evaluation.

## Note: Pixi Tasks Build Binaries Automatically

The pixi benchmark tasks automatically build binaries with `mojo build` before running (via `depends-on`). This avoids JIT compilation overhead that would skew results.

```bash
# Pixi tasks handle the build automatically
pixi run bench-gpu benchmark/datasets/twitter_large_record.json

# First run may have GPU initialization overhead - subsequent runs are faster
```

## Benchmarking Methodology

### What We Measure

`pixi run bench-gpu` emits a single `std.benchmark.Bench` report with four
rows so you can see where time goes across the GPU pipeline:

| Row | What It Includes | Use Case |
|-----|------------------|----------|
| **from host bytes: memcpy + parse (wall-clock)** | host→pinned memcpy + `parse_json_gpu_from_pinned` | Realistic steady-state "I have N bytes in memory, parse them via GPU" |
| **parse_json_gpu_from_pinned (pinned, wall-clock)** | H2D + GPU kernels + stream compaction + D2H + CPU bracket matching | Apples-to-apples comparison with cuJSON (both assume pinned input) |
| **parse_json_gpu_from_pinned (device-only)** | Same call, timed via `DeviceContext.execution_time` (CUDA events) | Pure device-queue time, excludes host-side CPU post-processing |
| **loads[target='gpu']** | Everything + `Value` tree construction on CPU | Real-world application performance |

Pass `--debug-timing` to get a per-phase breakdown inside each
`parse_json_gpu*` call (H2D, GPU kernels, position extraction, bracket
matching, total). The flag is a runtime argv parse, no recompile needed.

```bash
pixi run bench-gpu -- --debug-timing benchmark/datasets/twitter_large_record.json
```

### Apples-to-Apples Comparison with cuJSON

[cuJSON](https://github.com/AutomataLab/cuJSON) is the state-of-the-art GPU JSON parser from the academic literature. Our benchmark uses the exact same scope as cuJSON's benchmark to ensure fair comparison.

#### What Both Benchmarks Measure

| Step | cuJSON | json |
|------|--------|--------|
| **Input memory** | Pinned (cudaMallocHost) | Pinned (HostBuffer) |
| **H2D transfer** | ✓ (copy to GPU) | ✓ (copy to GPU) |
| **GPU processing** | Validation + Tokenization + Parser | Quote detection + Prefix sums + Bitmap + Stream compaction |
| **Bracket matching** | GPU (Parser kernel) | CPU (stack algorithm) |
| **D2H transfer** | ✓ (465MB structural data) | ✓ (4MB position indices) |
| **Output** | Structural positions + bracket pairs | Structural positions + bracket pairs |

Both parsers produce the same output: an array of structural character positions and their corresponding bracket pair mappings. This is what's needed for downstream JSON tree construction.

#### Detailed Timing Breakdown (804MB file)

```
cuJSON (182ms total):                json pinned (103ms total):
├─ H2D transfer:     15.2 ms         ├─ H2D transfer:      ~15 ms
├─ Validation:        1.5 ms         ├─ GPU kernels:       ~25 ms
├─ Tokenization:      5.5 ms         │  ├─ Quote detection
├─ Parser (GPU):      1.4 ms         │  ├─ Prefix sums
└─ D2H transfer:    158.6 ms         │  └─ Structural bitmap
                                      ├─ Stream compaction: ~45 ms (GPU)
                                      ├─ D2H transfer:      ~10 ms (4MB)
                                      └─ Bracket matching:  ~10 ms (CPU)
────────────────────────────         ────────────────────────────────
Throughput: 4.6 GB/s                 Throughput: 8.2 GB/s
```

### Why json is Faster

The **1.8x speedup** comes primarily from **GPU stream compaction**:

- **cuJSON approach:** Transfer all structural character data back to CPU
  - Structural chars = ~58% of input = 465MB for 804MB file
  - D2H transfer time: ~160ms

- **json approach:** Use GPU stream compaction to extract only positions
  - Position array = ~1M positions × 4 bytes = 4MB
  - D2H transfer time: ~10ms

**Speedup:** 16x reduction in D2H transfer size → 3.2x faster D2H → 1.8x overall speedup

### What About the "from host bytes" Row?

The "from host bytes" row adds the realistic cost of getting your bytes
onto the GPU. It reuses a long-lived `DeviceContext` and pinned
`HostBuffer` across iterations (as any real application would) and
measures `memcpy(pinned <- host)` + `parse_json_gpu_from_pinned`:

| Metric | Time | Throughput | Notes |
|--------|------|------------|-------|
| cuJSON (from pinned) | 182 ms | 4.6 GB/s | Assumes input is already pinned |
| json pinned (wall-clock) | 103 ms | 8.2 GB/s | Same assumption (fair comparison) |
| json "from host bytes" | ~280 ms | ~2.9 GB/s | Realistic scenario with host→pinned memcpy |

The host→pinned copy is the dominant extra cost (~100-150 ms for 804 MB
on DDR5). In practice you can avoid it by:
1. Reading files directly into a `HostBuffer` (pinned memory)
2. Memory-mapping files into pinned memory
3. Using network/RDMA buffers that are already pinned

### End-to-End Performance

For real applications using the full `loads[target='gpu']()` API:

| Pipeline Stage | Time (804 MB) |
|----------------|---------------|
| `parse_json_gpu_from_pinned` | ~150 ms |
| host→pinned memcpy | ~120 ms |
| Value tree construction (CPU) | ~600 ms |
| **Total** | **~900 ms** |
| **Throughput** | **~1.0 GB/s** |

The `Value` tree construction is currently CPU-bound; it's the largest
slice of the full-pipeline budget. Use `parse_json_gpu_from_pinned`
directly if you only need the structural position array and can build
your own downstream representation.

## Running Benchmarks

### GPU Benchmarks

```bash
# Compare json GPU vs cuJSON (recommended)
pixi run bench-gpu-cujson benchmark/datasets/twitter_large_record.json

# json GPU only (with detailed timing breakdown)
pixi run bench-gpu benchmark/datasets/twitter_large_record.json

# Try different datasets
pixi run bench-gpu-cujson benchmark/datasets/walmart_large_record.json
```

### CPU Benchmarks

```bash
# Compare json CPU vs native simdjson
pixi run bench-cpu benchmark/datasets/twitter.json

# Try different datasets
pixi run bench-cpu benchmark/datasets/citm_catalog.json
```

## Datasets

### Included (small, for quick tests)

Committed to the repository in `benchmark/datasets/`:

| File | Size | Source |
|------|------|--------|
| `twitter.json` | 632 KB | [simdjson](https://github.com/simdjson/simdjson) |
| `citm_catalog.json` | 1.6 MB | [simdjson](https://github.com/simdjson/simdjson) |

### Large Datasets (download required)

For GPU benchmarks, download large files from [cuJSON's Google Drive](https://drive.google.com/drive/folders/1PkDEy0zWOkVREfL7VuINI-m9wJe45P2Q):

Use pixi tasks (gdown is included in dev dependencies):

```bash
cd benchmark/datasets

# twitter_large_record.json (804MB) - PRIMARY BENCHMARK FILE
gdown 1mdF4HT7s0Jp4XZ0nOxY7lQpcwRZzCjE1 -O twitter_large_record.json

# walmart_large_record.json (950MB)
gdown 10vicgS7dPa4aL5PwEjqAvpAKCXYLblMt -O walmart_large_record.json

# wiki_large_record.json (1.1GB)
gdown 1bXdzhfWSdrnpg9WKOeV-oanYIT2j4yLE -O wiki_large_record.json
```

Install `gdown` if needed: `pip install gdown`

## Reproducibility

### cuJSON Version

Published results are against cuJSON at commit
`2ac7d3dcd7ad1ff64ebdb14022bf94c59b3b4953` from
[AutomataLab/cuJSON](https://github.com/AutomataLab/cuJSON). Pin your
clone to that commit for byte-exact reproducibility:

```bash
cd benchmark/cuJSON
git checkout 2ac7d3dcd7ad1ff64ebdb14022bf94c59b3b4953
```

### Build Configuration

cuJSON is built with:

```bash
nvcc -O3 -w -std=c++17 -arch=sm_100 -o benchmark/cuJSON/build/cujson_benchmark \
  benchmark/cuJSON/paper_reproduced/src/cuJSON-standardjson.cu
```

**Note:** Adjust `-arch=sm_XX` for your GPU:
- `sm_100` for NVIDIA B200 (Blackwell)
- `sm_90` for NVIDIA H100 (Hopper)
- `sm_80` for NVIDIA A100 (Ampere)
- `sm_89` for RTX 4090 (Ada Lovelace)

### simdjson Version

simdjson is installed from conda-forge (`simdjson >=4.2.4,<5`, declared
in `pixi.toml`). The thin C++ FFI wrapper in `json/cpu/simdjson_ffi/` is
automatically built during `pixi install` via the activation hook.

## Hardware Requirements

### For GPU Benchmarks
- NVIDIA GPU with CUDA support (tested on B200, H100, A100)
- CUDA toolkit (latest version)
- At least 2GB GPU memory (for 804MB benchmark)

### For CPU Benchmarks
- Any modern CPU with AVX2 or better (simdjson requirement)
- 2GB+ RAM recommended

## Benchmark Code Structure

```
benchmark/
├── mojo/
│   ├── bench_cpu.mojo          # CPU: json (Mojo backend) via bench_function
│   ├── bench_backend.mojo      # json (Mojo backend) vs simdjson FFI
│   └── bench_gpu.mojo          # GPU: 4-row Bench report + --debug-timing
├── cpp/
│   └── bench_simdjson.cpp      # Native simdjson C++ reference
├── cuJSON/                      # Optional: clone AutomataLab/cuJSON here
│   └── build/
│       └── cujson_benchmark    # Built by pixi run -e dev build-cujson
├── datasets/
│   ├── twitter.json            # Small (632 KB, committed)
│   ├── citm_catalog.json       # Small (1.6 MB, committed)
│   └── *.json                  # Large files (download separately)
└── README.md                    # This file
```

## Performance Tips

### For Best GPU Performance

1. **Use pinned memory:** Pre-allocate with `HostBuffer` for H2D transfers
2. **Large files only:** GPU overhead dominates for files <1MB
3. **Warm-up runs:** First GPU kernel launch has one-time initialization overhead
4. **Batch processing:** Reuse `DeviceContext` across multiple parses

### For Best CPU Performance

1. **Small files:** CPU is faster than GPU for files <1MB
2. **Memory-mapped files:** For very large files, use memory mapping
3. **Batch processing:** Parse multiple files in parallel with threading

## Troubleshooting

### "CUDA out of memory"
- Reduce file size or use CPU backend
- Close other GPU applications
- Check available GPU memory with `nvidia-smi`

### "cuJSON benchmark not found"
- Run `pixi run -e dev build-cujson` first (the task lives in the dev feature)
- Check that CUDA toolkit is installed

### "Failed to create DeviceContext"
- Verify GPU is available: `nvidia-smi`
- Check CUDA installation
- Ensure Mojo has GPU support (nightly build)

## Further Reading

- [Performance Deep Dive](../docs/performance.md) - Detailed optimization explanations
- [Architecture Overview](../docs/architecture.md) - System design and pipeline details
- [cuJSON Paper](https://arxiv.org/abs/2109.07569) - Academic reference for GPU JSON parsing
