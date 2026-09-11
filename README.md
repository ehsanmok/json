# json

[![CI](https://github.com/ehsanmok/json/actions/workflows/ci.yml/badge.svg)](https://github.com/ehsanmok/json/actions/workflows/ci.yml)
[![Docs](https://github.com/ehsanmok/json/actions/workflows/docs.yaml/badge.svg)](https://ehsanmok.github.io/json/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**High-performance JSON for Mojo** 🔥 Pure-Mojo two-pass CPU parser, GPU-accelerated parsing on NVIDIA, AMD, and Apple Metal, tape-backed `Document` shared by every backend, reflection serde with zero boilerplate, RFC-compliant JSONPath, JSON Patch, and JSON Schema. The simdjson FFI shim is opt-in for the cases where you need it.

```mojo
from json import loads, dumps

var data = loads('{"name": "Alice", "scores": [95, 87, 92]}')
print(data["name"].string_value())     # Alice
print(data["scores"][0].int_value())   # 95
print(dumps(data, indent="  "))        # pretty print
```

## Why json

- **Pure Mojo, zero FFI on the hot path.** The default CPU parser is a 64-byte branchless SIMD scan (PSHUFB-style classifier with prefix-XOR escape tracking) that emits a packed `Document` tape. The simdjson FFI shim is opt-in via `target="cpu-simdjson"`.
- **One representation across CPU and GPU.** Every backend writes into the same tape-backed `Document`. `Value` is a stable index into that tape, so iteration is a tape walk rather than a re-parse, and nested mutation propagates through the parent (`doc["a"]["b"].set(...)` is observed by `doc`).
- **GPU that wins on big files.** [Numbers below](#performance); details in [`docs/performance.md`](./docs/performance.md).
- **Reflection serde with no boilerplate.** `serialize_json(struct)` and `deserialize_json[T](json)` walk struct fields at compile time, no hand-written `to_json` / `from_json` needed. Custom traits (`JsonSerializable`, `JsonDeserializable`) override the default for one type without abandoning reflection for the rest.
- **Strict where it matters, lenient where it asks for it.** RFC 7159 by default; opt into comments, trailing commas, and a custom max-depth via `ParserConfig`.
- **Fuzzed.** Five mozz harnesses (parser, simdjson FFI, Value access, JSONPath, NDJSON) with a differential property that simdjson and the native parser must agree on canonical `dumps` output, plus an ASan harness over the FFI and tape boundaries.

## Install

```toml
[workspace]
channels = ["https://conda.modular.com/max-nightly", "conda-forge"]
preview = ["pixi-build"]

[dependencies]
json = { git = "https://github.com/ehsanmok/json.git", tag = "<latest-release>" }
```

```bash
pixi install
```

Requires [pixi](https://pixi.sh). Pin to a [released tag](https://github.com/ehsanmok/json/releases) for reproducible builds; track unreleased work via `branch = "main"` (breaking changes possible between tags).

`mojo` and `simdjson` install as transitive dependencies. The simdjson FFI wrapper builds on environment activation, no manual step.

For GPU acceleration: NVIDIA CUDA 7.0+, AMD ROCm 6+, or Apple Silicon. See [GPU compatibility](https://docs.modular.com/max/packages#gpu-compatibility).

## Quick start

```mojo
from json.prelude import *  # loads, dumps, load, dump, Value, Null, ParserConfig, SerializerConfig, ...

def main() raises:
    var data = loads('{"name": "Alice", "scores": [95, 87, 92]}')
    print(data["name"].string_value())     # Alice
    print(data["scores"][0].int_value())   # 95
    print(dumps(data, indent="  "))        # pretty print

    var config = load("config.json")
    var logs   = load[format="ndjson"]("events.ndjson")  # List[Value]
```

`json.prelude` re-exports the everyday surface (`loads`, `dumps`, `load`, `dump`, `Value`, `Null`, `ParserConfig`, `SerializerConfig`, the reflection serde shortcuts). Domain-specific surfaces (jsonpath, patch, schema, lazy, streaming, manual serde traits, simdjson FFI) stay in their own modules. Full list: [`json/prelude.mojo`](./json/prelude.mojo).

Typed serde via compile-time reflection (no hand-written `to_json` / `from_json`), GPU parsing for big files, NDJSON, JSONPath, JSON Patch, and JSON Schema each get a dedicated worked example under [`examples/`](./examples/) -- one example per topic, indexed in [`examples/README.md`](./examples/README.md). Every example runs as part of `pixi run tests`.

Supported reflection field types include `Int`, `Int64`, `Bool`, `Float64`, `Float32`, `String`, `List[T]`, `Optional[T]`, nested structs, `Value` (raw passthrough), `Dict[String, T]`, and nested `List` / `Optional` combinations. For full control on a single type, implement `JsonSerializable` / `JsonDeserializable`; the rest of the struct keeps using reflection. See the [API reference](https://ehsanmok.github.io/json/).

## Performance

### GPU on `twitter_large_record.json` (804 MB)

| Platform | Throughput | Pipeline |
|---|---:|---|
| AMD MI355X | 13 GB/s | lean (single-shot) |
| NVIDIA B200 | 6.5 GB/s | lean (single-shot, pinned wall-clock) |
| Apple M3 Pro | 3.1 GB/s | lean Metal (chunked at 64 MB) |

NVIDIA / AMD / Apple all run the same lean pipeline -- a single fused kernel + positions-only stream compaction. The CPU tape adapter consumes only `gpu_result.structural` and walks the byte stream once to apply the in-string escape state machine, so no `pair_pos` array or popcount + hierarchical prefix-sum cascade is needed. GPU is only beneficial for files >100 MB on discrete cards.

### CPU on x86 (Mojo native two-pass)

| File | Size | `parse_only` | `parse_traverse` |
|---|---:|---:|---:|
| `twitter.json` | 616 KB | 0.598 ms / 1.06 GB/s | 0.985 ms / 0.64 GB/s |
| `citm_catalog.json` | 1.7 MB | 1.109 ms / 1.56 GB/s | 1.980 ms / 0.87 GB/s |

Reproduce with `pixi run -e dev bench-cpu <file>` (3 warmup + 100 measured iterations, min-time-derived throughput). `parse_traverse` only adds a small constant on top of `parse_only` because every `Value` is a stable tape index, so traversal is a tape walk and not a re-parse. The gap to native simdjson on `parse_only` is algorithmic (no Eisel-Lemire float fast path, no AVX-512 64-byte chunks). Full breakdown in [`docs/performance.md`](./docs/performance.md).

### Serialization (write path)

Same fixture and same run as a like-for-like comparison against two other
Mojo JSON libraries -- the `document` shape from
[GLD.SerializerBenchmark](https://github.com/leo-gan/GLD.SerializerBenchmark),
100 records, ~47 KB, best of 15 on an Apple M-series host:

| Path | Serialize |
|---|---:|
| `serialize_json(struct)` -- typed, no intermediate tree | 37-40 us |
| EmberJson `serialize(struct)` -- typed reflection | 32-35 us |
| mojo-json -- schema-hardcoded writer | 25-26 us |
| `dumps(value)` after building a `Value` tree | ~1500 us |

Two things worth reading off that table. The typed path is roughly 22x
faster than routing the same data through a `Value` tree, so when the
shape is known ahead of time, use `serialize_json` (or write into a
`JsonWriter` directly) rather than assembling a document. And the
remaining gap to `mojo-json` is paid for: its decoder asserts fixed byte
offsets and rejects whitespace, reordered object members and unknown
members -- all of which RFC 8259 permits and this library accepts -- and
its float writer is not round-trip exact.

Building a `Value` tree is linear, but each `set` / `append` deep-copies
the subtree it is given unless you hand over ownership. For large trees,
transfer with `^`:

```mojo
var items = Value.array()
for it in rows:
    items.append(build_row(it))   # temporary: moved, no copy
o.set("items", items^)            # named local: `^` avoids a deep copy
```

Reproduce with `pixi run -e dev bench-build`.

Bench binaries are built with `mojo build -D ASSERT=none` so the Mojo stdlib's safety asserts are stripped, matching simdjson C++'s `-O3` posture for apples-to-apples comparison. `pixi run -e dev bench-cpu` and `bench-gpu` already pass this flag; the default `ASSERT=safe` build keeps the asserts in for development and costs ~20-37% on these workloads.

```bash
pixi run -e dev download-twitter-large                                # optional: cuJSON dataset
pixi run -e dev bench-cpu                                             # twitter.json
pixi run -e dev bench-cpu benchmark/datasets/citm_catalog.json
pixi run bench-gpu benchmark/datasets/twitter_large_record.json
pixi run -e dev bench-gpu-apple                                       # Apple Metal sweep
```

> **Apple Metal users on Xcode 26.x.** If `bench-gpu` errors with `Metal Compiler failed to compile metallib`, run `xcodebuild -downloadComponent MetalToolchain` once to register the toolchain. See [the Apple developer thread](https://developer.apple.com/forums/thread/802155) for context.

## Architecture

Every backend (Mojo native CPU, simdjson FFI, GPU) emits the same `Document` tape, so `Value` access, mutation, JSONPath, JSON Patch, and Schema validation see one DOM regardless of how the bytes arrived. Full request lifecycle, tape layout, module-by-module breakdown, and `Value` semantics in [`docs/architecture.md`](./docs/architecture.md).

## Examples

Examples are organised by tier (basic / intermediate / advanced); see [`examples/README.md`](./examples/README.md) for the guided tour. Every example runs as part of `pixi run tests`.

## Develop

```bash
git clone https://github.com/ehsanmok/json.git && cd json
pixi install                  # default env: tests, examples, library
pixi install -e dev           # adds mojodoc, pre-commit, dataset downloader
```

The project uses three pixi environments, layered:

| Env | Adds | What it unlocks |
|---|---|---|
| `default` | nothing | `tests-cpu`, `tests-gpu`, `tests-e2e`, `examples`, `format-check` |
| `dev` | mojodoc, pre-commit, gdown, sysroot pin, gxx | `format`, `docs`, `bench-cpu`, `bench-gpu`, `tests-asan`, `download-*` |
| `fuzz` | mozz on top of `dev` | `fuzz-loads`, `fuzz-simdjson`, `fuzz-value-access`, `fuzz-jsonpath`, `fuzz-ndjson`, `fuzz-all` |

Common tasks (run with `pixi run [-e <env>] <task>`):

| Task | Env | What it does |
|---|---|---|
| `tests` | default | Full unit + integration suite plus every example under [`examples/`](./examples/) |
| `tests-cpu` / `tests-gpu` / `tests-e2e` | default | Tier-specific suites |
| `tests-asan` | dev | LLVM AddressSanitizer over the FFI and tape boundaries (Linux) -- see [`pixi.toml`](./pixi.toml). |
| `format-check` / `format` | default / dev | `mojo format` over `json`, `tests`, `benchmark/mojo` |
| `docs` / `docs-build` | dev | mojodoc-rendered package docstring |
| `bench-cpu` / `bench-gpu` / `bench-gpu-apple` | dev | Reproducible benches |
| `fuzz-all` | fuzz | Every mozz harness back-to-back (parser, simdjson FFI, Value access, JSONPath, NDJSON) -- see [`fuzz/`](./fuzz/) + the `[feature.fuzz.tasks]` block in [`pixi.toml`](./pixi.toml). |

The full task list, including every per-example and per-fuzz target, is in [`pixi.toml`](./pixi.toml).

## Further reading

- [API reference (mojodoc)](https://ehsanmok.github.io/json/): full public API, auto-generated from docstrings.
- [`docs/architecture.md`](./docs/architecture.md): CPU and GPU backend design, tape layout, `Value` semantics.
- [`docs/performance.md`](./docs/performance.md): benchmark methodology, optimisation deep-dive, what's left to close the simdjson gap.
- [`benchmark/README.md`](./benchmark/README.md): reproducible benchmark setup.
- [`examples/README.md`](./examples/README.md): guided tour of basic, intermediate, advanced examples.

## License

[MIT](./LICENSE)
