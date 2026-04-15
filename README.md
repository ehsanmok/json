# json

[![CI](https://github.com/ehsanmok/json/actions/workflows/ci.yml/badge.svg)](https://github.com/ehsanmok/json/actions/workflows/ci.yml)
[![Docs](https://github.com/ehsanmok/json/actions/workflows/docs.yaml/badge.svg)](https://ehsanmok.github.io/json/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance JSON library for Mojo with GPU acceleration.

- **Python-like API:** `loads`, `dumps`, `load`, `dump`
- **Reflection serde:** zero-boilerplate struct serialization via compile-time reflection
- **GPU accelerated:** 2-4x faster than [cuJSON](https://github.com/AutomataLab/cuJSON) on large files
- **Cross-platform:** NVIDIA, AMD, and Apple Silicon GPUs
- **Streaming and lazy parsing:** handle files larger than memory
- **JSONPath and Schema:** query and validate JSON documents
- **RFC compliant:** JSON Patch, Merge Patch, JSON Pointer

## Quick Start

```mojo
from json import loads, dumps, load, dump

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

## Installation

Add json to your project's `pixi.toml`:

```toml
[workspace]
channels = ["https://conda.modular.com/max-nightly", "conda-forge"]
preview = ["pixi-build"]

[dependencies]
json = { git = "https://github.com/ehsanmok/json.git", branch = "main" }
```

Then run:

```bash
pixi install
```

Requires [pixi](https://pixi.sh) (pulls Mojo nightly automatically).

> **Note:** `mojo-compiler` and `simdjson` are automatically installed as dependencies.

**GPU (optional):** NVIDIA CUDA 7.0+, AMD ROCm 6+, or Apple Silicon. See [GPU requirements](https://docs.modular.com/max/packages#gpu-compatibility).

## Performance

### GPU (804MB `twitter_large_record.json`)

| Platform | Throughput | vs cuJSON |
|----------|------------|-----------|
| AMD MI355X | 13 GB/s | **3.6x faster** |
| NVIDIA B200 | 8 GB/s | **1.8x faster** |
| Apple M3 Pro | 3.9 GB/s | N/A |

*GPU only beneficial for files >100MB.*

```bash
# Download large dataset first (required for meaningful GPU benchmarks)
pixi run download-twitter-large

# Run GPU benchmark (only use large files)
pixi run bench-gpu benchmark/datasets/twitter_large_record.json
```

## Reflection-Based Serde (Zero Boilerplate)

Automatically serialize and deserialize structs using compile-time reflection. No hand-written `to_json()` or `from_json()` methods needed.

```mojo
from json import serialize_json, deserialize_json

@fieldwise_init
struct Person(Defaultable, Movable):
    var name: String
    var age: Int
    var active: Bool
    def __init__(out self):
        self.name = ""
        self.age = 0
        self.active = False

# Serialize: one function, zero boilerplate
var json = serialize_json(Person(name="Alice", age=30, active=True))
# {"name":"Alice","age":30,"active":true}

# Deserialize: just specify the type
var person = deserialize_json[Person](json)
print(person.name)  # Alice

# Pretty print
print(serialize_json[pretty=True](person))

# GPU-accelerated parsing, CPU struct extraction
var fast = deserialize_json[Person, target="gpu"](json)

# Non-raising variant (returns Optional)
from json import try_deserialize_json
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
from json import JsonSerializable, JsonDeserializable

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

Full API reference: [ehsanmok.github.io/json](https://ehsanmok.github.io/json/)

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

## Development

```bash
git clone https://github.com/ehsanmok/json.git && cd json
pixi install
pixi run tests-cpu
```

Further documentation:

- [Architecture](./docs/architecture.md): CPU/GPU backend design
- [Performance](./docs/performance.md): optimization deep dive
- [Benchmarks](./benchmark/README.md): reproducible benchmarks

## License

[MIT](./LICENSE)
