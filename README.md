# High-Performance JSON library for Mojo🔥

- **Python-like API** — `loads`, `dumps`, `load`, `dump`
- **GPU accelerated** — 2-4x faster than [cuJSON](https://github.com/AutomataLab/cuJSON) on large files
- **Cross-platform** — NVIDIA, AMD, and Apple Silicon GPUs
- **Streaming & lazy parsing** — Handle files larger than memory
- **JSONPath & Schema** — Query and validate JSON documents
- **RFC compliant** — JSON Patch, Merge Patch, JSON Pointer

## Quick Start

```bash
git clone --recursive https://github.com/user/mojson.git && cd mojson
pixi install
pixi run tests-cpu
```

```mojo
from mojson import loads, dumps, load, dump

# Parse & serialize strings
var data = loads('{"name": "Alice", "scores": [95, 87, 92]}')
print(data["name"].string_value())  # Alice
print(data["scores"][0].int_value())  # 95
print(dumps(data, indent="  "))  # Pretty print

# File I/O (auto-detects .ndjson)
var config = load("config.json")
var logs = load("events.ndjson")  # Returns array of values

# Explicit GPU parsing
var big = load[target="gpu"]("large.json")
```

## Requirements

- [Mojo/MAX SDK](https://docs.modular.com/max/packages)
- [pixi](https://pixi.sh) package manager

**GPU (optional):** NVIDIA CUDA 7.0+, AMD ROCm 6+, or Apple Silicon. See [GPU requirements](https://docs.modular.com/max/packages#gpu-compatibility).

## Performance

| Platform | Throughput | vs cuJSON |
|----------|------------|-----------|
| AMD MI355X | 13 GB/s | **3.6x faster** |
| NVIDIA B200 | 7 GB/s | **2x faster** |
| Apple M3 Pro | 3.3 GB/s | — |
| CPU (simdjson) | 4.5 GB/s | — |

*Benchmarks on 804MB JSON file. GPU recommended for files >100MB.*

```bash
pixi run bench-gpu benchmark/datasets/twitter.json
pixi run bench-gpu-all  # All datasets, 3 runs each
```

## API

Everything through 4 functions: `loads`, `dumps`, `load`, `dump`

```mojo
# Parse strings (default: CPU)
loads(s)                              # JSON string -> Value
loads[target="gpu"](s)                # GPU parsing
loads[format="ndjson"](s)             # NDJSON string -> List[Value]
loads[lazy=True](s)                   # Lazy parsing (CPU only)

# Serialize strings
dumps(v)                              # Value -> JSON string
dumps(v, indent="  ")                 # Pretty print
dumps[format="ndjson"](values)        # List[Value] -> NDJSON string

# File I/O (auto-detects .ndjson from extension)
load("data.json")                     # JSON file -> Value (CPU)
load("data.ndjson")                   # NDJSON file -> Value (array)
load[target="gpu"]("large.json")      # GPU parsing
load[streaming=True]("huge.ndjson")   # Stream (CPU, memory efficient)
dump(v, f)                            # Write to file

# Value access
value["key"], value[0]                # By key/index
value.at("/path")                     # JSON Pointer (RFC 6901)
value.set("key", val)                 # Mutation

# Advanced
jsonpath_query(doc, "$.users[*]")     # JSONPath queries
validate(doc, schema)                 # JSON Schema validation
apply_patch(doc, patch)               # JSON Patch (RFC 6902)
```

### Feature Matrix

| Feature | CPU | GPU | Notes |
|---------|-----|-----|-------|
| `loads(s)` | ✅ default | ✅ `target="gpu"` | |
| `load(path)` | ✅ default | ✅ `target="gpu"` | Auto-detects .ndjson |
| `loads[format="ndjson"]` | ✅ default | ✅ `target="gpu"` | |
| `loads[lazy=True]` | ✅ | — | CPU only |
| `load[streaming=True]` | ✅ | — | CPU only |
| `dumps` / `dump` | ✅ | — | CPU only |

Full API: [docs/api.md](./docs/api.md)

## Examples

```bash
pixi run mojo -I . examples/01_basic_parsing.mojo
```

| Example | Description |
|---------|-------------|
| [01_basic_parsing](./examples/01_basic_parsing.mojo) | Parse, serialize, type handling |
| [02_file_operations](./examples/02_file_operations.mojo) | Read/write JSON files |
| [03_value_types](./examples/03_value_types.mojo) | Type checking, value extraction |
| [04_gpu_parsing](./examples/04_gpu_parsing.mojo) | GPU-accelerated parsing |
| [05_error_handling](./examples/05_error_handling.mojo) | Error handling patterns |
| [06_struct_serde](./examples/06_struct_serde.mojo) | Struct serialization |
| [07_ndjson](./examples/07_ndjson.mojo) | NDJSON parsing & streaming |
| [08_lazy_parsing](./examples/08_lazy_parsing.mojo) | On-demand lazy parsing |
| [09_jsonpath](./examples/09_jsonpath.mojo) | JSONPath queries |
| [10_schema_validation](./examples/10_schema_validation.mojo) | JSON Schema validation |
| [11_json_patch](./examples/11_json_patch.mojo) | JSON Patch & Merge Patch |

## Documentation

- [API Reference](./docs/api.md) — Complete function reference
- [Architecture](./docs/architecture.md) — CPU/GPU backend design
- [Performance](./docs/performance.md) — Optimization deep dive
- [Benchmarks](./benchmark/README.md) — Reproducible benchmarks

## License

[MIT](./LICENSE)
