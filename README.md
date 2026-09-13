# json

[![CI](https://github.com/ehsanmok/json/actions/workflows/ci.yml/badge.svg)](https://github.com/ehsanmok/json/actions/workflows/ci.yml)
[![Docs](https://github.com/ehsanmok/json/actions/workflows/docs.yaml/badge.svg)](https://ehsanmok.github.io/json/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**High-performance JSON for Mojo** 🔥 Pure-Mojo two-pass CPU parser, GPU-accelerated parsing on NVIDIA, AMD, and Apple Metal, tape-backed `Document` shared by every backend, reflection serde with zero boilerplate, and JSONPath, JSON Pointer, JSON Patch and JSON Schema implemented against their specifications rather than approximated. The simdjson FFI shim is opt-in for the cases where you need it.

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
- **Strict where it matters, lenient where it asks for it.** RFC 8259 by default; opt into comments, trailing commas, and a custom max-depth via `ParserConfig`, or tighten to I-JSON (RFC 7493) with `ParserConfig.interoperable()`.
- **Conformance is a build gate, not a claim.** Catalogs for RFC 8259, RFC 7493, RFC 6901, RFC 6902, RFC 7396 and JSON Schema draft 2020-12 run in `pixi run tests-cpu`. Each runner asserts the set of failing cases equals a declared list of known gaps, so a regression fails the build and so does fixing a gap without deleting its entry. Those lists are empty.
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

### Typed serde (read and write path)

`serialize_json` and `deserialize_json` go straight between a struct
and JSON bytes: no intermediate `Value`, no tape. Five record shapes,
100 records each, median of seven calibrated batches on an Apple
M-series host (`pixi run -e dev bench-serde`):

| Shape | Bytes | `deserialize_json` | `loads` + `Value` walk | `serialize_json` |
|---|---:|---:|---:|---:|
| message | 16 KB | 15.5 us | 53.5 us | 13.3 us |
| document | 47 KB | 60.0 us | 192.9 us | 37.6 us |
| telemetry | 43 KB | 81.3 us | 148.9 us | 166.3 us |
| strings | 43 KB | 61.1 us | 114.2 us | 34.3 us |
| event | 27 KB | 39.1 us | 89.4 us | 15.4 us |

The middle column is what reading the same document costs if you route
it through `loads` and pick fields off `Value` by hand -- which is the
right tool when you are exploring a document, and the wrong one when
you already know its shape.

Telemetry serializes slowly: it is 32 floats per record, and shortest
round-trip float formatting is currently the most expensive thing this
library writes. That is a known gap rather than a property of the
design.

#### Against the other Mojo JSON libraries

On one machine, under one timing protocol, against the other two
pure-Mojo JSON libraries that build on Mojo 1.0.0:

- **Deserializing ordinary JSON -- any whitespace, any member order --
  this is the fastest of the three, on every shape, by 31% to 84%.**
  Its timings barely move when a document is reformatted (document:
  60.0 us canonical, 64.0 us with spaces and reversed members); the
  other two slow by 34% and 147%, and one of them cannot parse two of
  the shapes at all, because its list readers do not skip whitespace.
- Deserializing a compact document with members in a fixed order, the
  wire-format library is 20-126% faster. Its decoders match whole keys
  as machine words at known offsets, which is a good trade when you
  control both ends and own the byte layout.
- Serializing, this library leads on one shape of five.

Numbers are reproducible but not shipped: `benchmark/compare/` builds
its own environment and fetches the other libraries at pinned
versions. See [`benchmark/README.md`](./benchmark/README.md) for why it
is not in the tree, and that directory's own README for the version
pins and the ways in which the three are not doing identical work.

#### Building a `Value` tree

Tree construction is linear, but each `set` / `append` deep-copies the
subtree it is given unless you hand over ownership. For large trees,
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
