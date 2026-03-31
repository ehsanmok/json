# High-Performance JSON library for Mojo🔥

[![CI](https://github.com/ehsanmok/mojson/actions/workflows/ci.yml/badge.svg)](https://github.com/ehsanmok/mojson/actions/workflows/ci.yml)
[![Docs](https://github.com/ehsanmok/mojson/actions/workflows/docs.yaml/badge.svg)](https://github.com/ehsanmok/mojson/actions/workflows/docs.yaml)

- **Python-like API** — `loads`, `dumps`, `load`, `dump`
- **Reflection serde** — Zero-boilerplate struct serialization via compile-time reflection
- **GPU accelerated** — 2-4x faster than [cuJSON](https://github.com/AutomataLab/cuJSON) on large files
- **Cross-platform** — NVIDIA, AMD, and Apple Silicon GPUs
- **Streaming & lazy parsing** — Handle files larger than memory
- **JSONPath & Schema** — Query and validate JSON documents
- **RFC compliant** — JSON Patch, Merge Patch, JSON Pointer

## Requirements

[pixi](https://pixi.sh) package manager

**GPU (optional):** NVIDIA CUDA 7.0+, AMD ROCm 6+, or Apple Silicon. See [GPU requirements](https://docs.modular.com/max/packages#gpu-compatibility).

## Installation

Add mojson to your project's `pixi.toml`:

```toml
[workspace]
channels = ["https://conda.modular.com/max-nightly", "conda-forge"]
preview = ["pixi-build"]

[dependencies]
mojson = { git = "https://github.com/ehsanmok/mojson.git", branch = "main" }
```

Then run:

```bash
pixi install
```

> **Note:** `mojo-compiler` and `simdjson` are automatically installed as dependencies.

## Quick Start

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

## Development Setup

To contribute or run tests:

```bash
git clone https://github.com/ehsanmok/mojson.git && cd mojson
pixi install
pixi run tests-cpu
```

## Performance

### GPU (804MB `twitter_large_record.json`)

| Platform | Throughput | vs cuJSON |
|----------|------------|-----------|
| AMD MI355X | 13 GB/s | **3.6x faster** |
| NVIDIA B200 | 8 GB/s | **1.8x faster** |
| Apple M3 Pro | 3.9 GB/s | — |

*GPU only beneficial for files >100MB.*

```bash
# Download large dataset first (required for meaningful GPU benchmarks)
pixi run download-twitter-large

# Run GPU benchmark (only use large files)
pixi run bench-gpu benchmark/datasets/twitter_large_record.json
```

## Reflection-Based Serde (Zero Boilerplate)

Automatically serialize and deserialize structs using compile-time reflection — no hand-written `to_json()` or `from_json()` methods needed.

```mojo
from mojson import serialize_json, deserialize_json

@fieldwise_init
struct Person(Defaultable, Movable):
    var name: String
    var age: Int
    var active: Bool
    def __init__(out self):
        self.name = ""
        self.age = 0
        self.active = False

# Serialize — one function, zero boilerplate
var json = serialize_json(Person(name="Alice", age=30, active=True))
# {"name":"Alice","age":30,"active":true}

# Deserialize — just specify the type
var person = deserialize_json[Person](json)
print(person.name)  # Alice

# Pretty print
print(serialize_json[pretty=True](person))

# GPU-accelerated parsing, CPU struct extraction
var fast = deserialize_json[Person, target="gpu"](json)

# Non-raising variant (returns Optional)
from mojson import try_deserialize_json
var maybe = try_deserialize_json[Person]('{"bad json')  # None
```

### Supported Field Types

| Category | Types |
|----------|-------|
| Scalars | `Int`, `Int64`, `Bool`, `Float64`, `Float32`, `String` |
| Containers | `List[T]`, `Optional[T]` (where T is a scalar) |
| Nested | Any struct that is `Defaultable & Movable` |
| Raw JSON | `Value` (pass-through, no conversion) |

### Custom Serialization

Override reflection behavior for specific types:

```mojo
from mojson import JsonSerializable, JsonDeserializable

struct Color(JsonSerializable, Defaultable, Movable):
    var r: Int
    var g: Int
    var b: Int

    def to_json_value(self) raises -> Value:
        """Serialize as "rgb(r,g,b)" instead of {"r":...,"g":...,"b":...}."""
        ...

struct RGBArray(JsonDeserializable, Defaultable, Movable):
    var r: Int
    var g: Int
    var b: Int

    @staticmethod
    def from_json_value(json: Value) raises -> Self:
        """Deserialize from JSON array [r, g, b] instead of object."""
        ...
```

### Reflection Serde API

| Function | Description |
|----------|-------------|
| `serialize_json(value)` | Struct → JSON string |
| `serialize_json[pretty=True](value)` | Struct → pretty JSON string |
| `serialize_value(value)` | Struct → `Value` object |
| `deserialize_json[T](json_str)` | JSON string → struct `T` |
| `deserialize_json[T, target="gpu"](s)` | GPU parse → struct `T` |
| `deserialize_value[T](value)` | `Value` object → struct `T` |
| `try_deserialize_json[T](json_str)` | Non-raising, returns `Optional[T]` |

## API

Everything through 4 functions: `loads`, `dumps`, `load`, `dump`

```mojo
# Parse strings (default: pure Mojo backend - fast, zero FFI)
loads(s)                              # JSON string -> Value
loads[target="cpu-simdjson"](s)       # Use simdjson FFI backend
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

### CPU Backends

| Backend | Target | Speed | Dependencies |
|---------|--------|-------|--------------|
| Mojo (native) | `loads()` (default) | **1.31 GB/s** | Zero FFI |
| simdjson (FFI) | `loads[target="cpu-simdjson"]()` | 0.48 GB/s | libsimdjson |

The pure Mojo backend is the **default** and is **~2.7x faster** than the FFI approach with zero external dependencies.

Full API: [ehsanmok.github.io/mojson](https://ehsanmok.github.io/mojson/)

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
| [06_struct_serde](./examples/06_struct_serde.mojo) | Struct serialization (manual traits) |
| [07_ndjson](./examples/07_ndjson.mojo) | NDJSON parsing & streaming |
| [08_lazy_parsing](./examples/08_lazy_parsing.mojo) | On-demand lazy parsing |
| [09_jsonpath](./examples/09_jsonpath.mojo) | JSONPath queries |
| [10_schema_validation](./examples/10_schema_validation.mojo) | JSON Schema validation |
| [11_json_patch](./examples/11_json_patch.mojo) | JSON Patch & Merge Patch |
| [12_reflection_serde](./examples/12_reflection_serde.mojo) | Zero-boilerplate reflection serde |

## Documentation

- [Architecture](./docs/architecture.md) — CPU/GPU backend design
- [Performance](./docs/performance.md) — Optimization deep dive
- [Benchmarks](./benchmark/README.md) — Reproducible benchmarks

## License

[MIT](./LICENSE)
