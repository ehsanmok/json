# Architecture

json provides a unified API with multiple high-performance backends: CPU (simdjson FFI or pure Mojo) and GPU (native Mojo kernels).

## System Overview

```mermaid
graph TB
    subgraph "json API"
        loads["loads(json_string)"]
        dumps["dumps(value)"]
    end

    subgraph "CPU Backend - Pure Mojo (Default)"
        mojo_parser["MojoJSONParser"]
        mojo_parse["Zero-FFI Parsing"]
    end

    subgraph "CPU Backend - simdjson FFI"
        simdjson["simdjson FFI"]
        cpu_parse["Parse & Build Value Tree"]
    end

    subgraph "GPU Backend"
        gpu_kernels["GPU Kernels"]
        stream_compact["Stream Compaction"]
        bracket_match["Bracket Matching"]
        tree_build["Value Tree Builder"]
    end

    loads -->|"default"| mojo_parser
    loads -->|"target='cpu-simdjson'"| simdjson
    loads -->|"target='gpu'"| gpu_kernels
    simdjson --> cpu_parse
    mojo_parser --> mojo_parse
    gpu_kernels --> stream_compact
    stream_compact --> bracket_match
    bracket_match --> tree_build
    cpu_parse --> dumps
    mojo_parse --> dumps
    tree_build --> dumps
```

## CPU Backends

### Pure Mojo Backend (Default)

**Implementation:** Native Mojo JSON parser with optimized parsing

**Location:**
- `json/cpu/mojo_backend.mojo` - MojoJSONParser struct
- `json/cpu/types.mojo` - Common JSON type constants

**Performance:** ~1.31 GB/s (on twitter.json)

**Usage:**
```mojo
from json import loads
var data = loads('{"key": "value"}')  # Default is Mojo backend
```

**Benefits:**
- Zero external dependencies (no libsimdjson required)
- 30% faster than FFI due to no marshalling overhead
- Easier deployment (single Mojo binary)

### simdjson FFI Backend

**Implementation:** FFI wrapper around [simdjson](https://github.com/simdjson/simdjson)

**Location:**
- `json/cpu/simdjson_ffi/` - C++ wrapper
- `json/cpu/simdjson_ffi.mojo` - Mojo FFI bindings

**Performance:** ~0.48 GB/s (on twitter.json)

**Usage:**
```mojo
from json import loads
var data = loads[target="cpu-simdjson"]('{"key": "value"}')
```

### CPU Parsing Flow (simdjson)

1. Load JSON string into memory
2. Call simdjson via FFI (`json/cpu/simdjson_ffi.mojo`)
3. Recursively build `Value` tree from simdjson result
4. Return parsed `Value`

### CPU Parsing Flow (Pure Mojo)

1. Copy JSON bytes to internal buffer
2. Recursive descent parsing with `MojoJSONParser`
3. Build `Value` tree directly
4. Return parsed `Value`

## GPU Backend

**Implementation:** Native Mojo GPU kernels inspired by [cuJSON](https://github.com/AutomataLab/cuJSON)

**Location:**
- `json/gpu/parser.mojo` - Main GPU parser (`parse_json_gpu`, `parse_json_gpu_from_pinned`)
- `json/gpu/kernels.mojo` - CUDA-style GPU kernels (fused bitmap + structural extraction)
- `json/gpu/stream_compact.mojo` - GPU stream compaction for position extraction
- `json/gpu/bracket_match.mojo` - GPU parallel bracket matching (experimental; the main parse path uses a CPU stack matcher after stream compaction)

**Performance:** ~8 GB/s on NVIDIA B200 (1.8x faster than cuJSON)

**Techniques:**
- Bitmap-based parsing
- Parallel prefix sums
- GPU stream compaction for position extraction
- Hybrid GPU/CPU pipeline

### GPU Pipeline

```mermaid
flowchart LR
    subgraph "Phase 1: Transfer"
        A[JSON Bytes] -->|H2D| B[GPU Memory]
    end

    subgraph "Phase 2: GPU Kernels"
        B --> C[Quote Detection]
        C --> D[Prefix Sum]
        D --> E[Structural Bitmap]
    end

    subgraph "Phase 3: Extract"
        E --> F[Stream Compaction]
        F --> G[Position Array]
    end

    subgraph "Phase 4: Build"
        G --> H[Bracket Matching]
        H --> I[Value Tree]
    end

    style A fill:#e1f5fe
    style I fill:#c8e6c9
```

### GPU Parsing Flow

1. **Host-to-Device Transfer:** Copy JSON bytes to GPU using pinned memory (HostBuffer) for fast transfer (~15ms for 804MB)
2. **GPU Kernels:** Execute parallel kernels to:
   - Create bitmaps for quotes, escapes, structural characters
   - Compute parallel prefix sums to identify in-string regions
   - Extract structural character bitmap
3. **Stream Compaction (GPU):** Extract only the positions of structural characters (~50ms)
4. **Device-to-Host Transfer:** Copy compact position array back to CPU
5. **Bracket Matching (CPU):** Match brackets using stack algorithm (~10ms)
6. **Value Tree Construction (CPU):** Build `Value` tree from structural info

### Why Hybrid GPU/CPU?

- **GPU excels at:** Parallel bitmap operations, prefix sums, stream compaction
- **CPU excels at:** Sequential bracket matching, tree construction with dynamic memory
- **Key insight:** GPU stream compaction dramatically reduces D2H transfer size (from 465MB to <10MB for 804MB input)

## Value Type

The `Value` struct represents any JSON value (null, bool, int, float, string, array, object).

See [API Reference](https://ehsanmok.github.io/json/) for complete `Value` methods.

## Directory Structure

```
json/
├── __init__.mojo              # Public API exports
├── parser.mojo                # Unified CPU/GPU parser, loads/load functions
├── serialize.mojo             # dumps/dump functions
├── value.mojo                 # Value type definition
├── types.mojo                 # JSONInput, JSONResult types
├── iterator.mojo              # JSONIterator for traversing results
├── ndjson.mojo                # NDJSON parsing/serialization
├── lazy.mojo                  # On-demand lazy parsing
├── streaming.mojo             # Streaming parser for large files
├── config.mojo                # Parser/Serializer configuration
├── errors.mojo                # Error formatting with line/column
├── unicode.mojo               # Unicode escape handling
├── patch.mojo                 # JSON Patch & Merge Patch (RFC 6902/7396)
├── jsonpath.mojo              # JSONPath query language
├── schema.mojo                # JSON Schema validation
├── reflection.mojo            # Compile-time reflection serde
├── deserialize.mojo           # serialize_json / deserialize_json API
├── cpu/
│   ├── __init__.mojo         # CPU backend exports
│   ├── types.mojo            # Common JSON type constants
│   ├── mojo_backend.mojo     # Pure Mojo JSON parser
│   ├── simd_backend.mojo     # SIMD-accelerated CPU parser
│   ├── simdjson_ffi.mojo     # simdjson FFI bindings
│   └── simdjson_ffi/         # C++ simdjson wrapper (libsimdjson via conda)
└── gpu/
    ├── parser.mojo            # GPU parser implementation
    ├── kernels.mojo           # GPU kernel functions
    ├── stream_compact.mojo    # GPU stream compaction
    └── bracket_match.mojo     # GPU parallel bracket-match (experimental)

tests/
├── test_api.mojo              # Unified API tests (loads/dumps/load/dump)
├── test_value.mojo            # Value type tests
├── test_parser.mojo           # Parser tests (simdjson backend)
├── test_mojo_backend.mojo     # Pure Mojo backend tests
├── test_serialize.mojo        # Serialization tests
├── test_serde.mojo            # Struct serialization tests
├── test_reflection.mojo       # Reflection-based serde tests
├── test_patch.mojo            # JSON Patch tests
├── test_jsonpath.mojo         # JSONPath tests
├── test_schema.mojo           # JSON Schema tests
├── test_e2e.mojo              # End-to-end tests
├── test_gpu.mojo              # GPU parser tests
├── test_gpu_kernels.mojo      # GPU kernel tests (stream compaction)
├── test_bracket_match.mojo    # GPU bracket-match tests
└── bench_bracket_match.mojo   # GPU bracket-match microbenchmark

benchmark/
├── datasets/                  # Benchmark files
├── mojo/
│   ├── bench_cpu.mojo        # CPU benchmark (simdjson FFI)
│   ├── bench_backend.mojo    # Backend comparison (simdjson vs Mojo)
│   └── bench_gpu.mojo        # GPU benchmark
├── cpp/
│   └── bench_simdjson.cpp    # Native simdjson C++ benchmark
└── cuJSON/                    # Optional cuJSON checkout (cloned manually;
                               # see benchmark/README.md) for head-to-head
```

## Build & Test

```bash
# Build simdjson FFI wrapper
pixi run build

# Run tests
pixi run tests-cpu  # CPU parser tests
pixi run tests-gpu  # GPU parser tests

# Benchmarks
pixi run bench-cpu   # CPU: json vs simdjson
pixi run bench-gpu   # GPU: json only
pixi run bench-gpu-cujson  # GPU: json vs cuJSON
```

## Dependencies

- **Mojo:** Latest nightly (with GPU support), pulled in automatically by `pixi install`
- **simdjson:** Installed from conda-forge (`simdjson >=4.2.4,<5`). The thin
  C++ FFI wrapper in `json/cpu/simdjson_ffi/` is auto-built by `pixi install`
  via the activation hook.
- **sysroot_linux-64:** `>=2.34` (Linux only) so `mojo build` can link
  against glibc 2.34 symbols referenced by Mojo's runtime libs.
- **cuJSON:** Optional; clone manually into `benchmark/cuJSON` for the
  head-to-head GPU benchmark. See `benchmark/README.md`.
- **CUDA:** Required for the GPU backend (any SM70+ NVIDIA GPU works;
  the library has also been tested on AMD ROCm and Apple Silicon).
