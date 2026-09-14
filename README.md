# json

[![CI](https://github.com/ehsanmok/json/actions/workflows/ci.yml/badge.svg)](https://github.com/ehsanmok/json/actions/workflows/ci.yml)
[![Docs](https://github.com/ehsanmok/json/actions/workflows/docs.yaml/badge.svg)](https://ehsanmok.github.io/json/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GPU: MAX Community License](https://img.shields.io/badge/GPU-MAX%20Community%20License-orange.svg)](json/gpu/LICENSE-GPU.md)

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
- **One representation across CPU and GPU.** Every backend writes into the same tape-backed `Document`. `Value` is a stable index into that tape, so iteration is a tape walk rather than a re-parse. A parsed value converts to an owned tree the first time it is mutated, once rather than once per write, and a nested write is addressed by pointer: `doc.set_at("/a/b", v)`.
- **GPU that wins on big files.** [Numbers below](#performance); details in [`docs/performance.md`](./docs/performance.md).
- **Reflection serde with no boilerplate.** `serialize_json(struct)` and `deserialize_json[T](json)` walk struct fields at compile time, no hand-written `to_json` / `from_json` needed. Custom traits (`JsonSerializable`, `JsonDeserializable`) override the default for one type without abandoning reflection for the rest.
- **Strict where it matters, lenient where it asks for it.** RFC 8259 by default; opt into comments, trailing commas, and a custom max-depth via `ParserConfig`, or tighten to I-JSON (RFC 7493) with `ParserConfig.interoperable()`.
- **Conformance is a build gate, not a claim.** [Every standard below](#standards-conformance) is checked by a catalog or a suite that runs in `pixi run tests-cpu`, which is what CI runs on Linux and macOS.
- **Fuzzed.** Five mozz harnesses (parser, simdjson FFI, Value access, JSONPath, NDJSON) with a differential property that simdjson and the native parser must agree on canonical `dumps` output, plus an ASan harness over the FFI and tape boundaries.

## Standards conformance

Every row runs in `pixi run tests-cpu`, which is the CI test job on both Linux and macOS. A catalog runner asserts that the set of failing cases equals a declared list of known gaps, so a regression fails the build and so does fixing a gap without deleting its entry. **Those lists are empty.**

| Standard | Checks | Result | Where |
|---|---:|---|---|
| [RFC 8259](https://datatracker.ietf.org/doc/html/rfc8259) JSON | 312 asserted, 35 informational | 0 unexpected failures | `tests/conformance/rfc8259.json` |
| [RFC 7493](https://datatracker.ietf.org/doc/html/rfc7493) I-JSON | 6 asserted | 0 unexpected failures | `tests/conformance/rfc7493-ijson.json` |
| [RFC 6901](https://datatracker.ietf.org/doc/html/rfc6901) JSON Pointer | 18 asserted | 0 unexpected failures | `tests/conformance/rfc6901-pointer.json` |
| [RFC 6902](https://datatracker.ietf.org/doc/html/rfc6902) JSON Patch | 27 asserted | 0 unexpected failures | `tests/conformance/rfc6902-patch.json` |
| [RFC 7396](https://datatracker.ietf.org/doc/html/rfc7396) Merge Patch | 21 asserted, 3 informational | 0 unexpected failures | `tests/conformance/rfc7396-merge-patch.json` |
| [JSON Schema 2020-12](https://json-schema.org/draft/2020-12) | 197 asserted | 0 unexpected failures, 44 of 44 keywords covered | `tests/conformance/jsonschema-2020-12.json` |
| [RFC 9535](https://datatracker.ietf.org/doc/html/rfc9535) JSONPath | 80 tests | all pass | `tests/test_jsonpath.mojo` |
| [RFC 9485](https://datatracker.ietf.org/doc/html/rfc9485) I-Regexp | 32 tests | all pass | `tests/test_regex.mojo` |

The six catalogs are transcribed from the specification texts, each case carrying the section it came from. RFC 8259's corpus also merges [JSONTestSuite](https://github.com/nst/JSONTestSuite); see [`tests/conformance/README.md`](./tests/conformance/README.md) for provenance. The two RFCs checked by unit tests rather than a catalog use the specifications' own worked examples: the JSONPath tests are organised by section, including the section 1.5 bookstore queries and about seventy queries that must be rejected.

Deviations are recorded rather than hidden. The informational rows are cases the RFC states at SHOULD level and this library decides differently; each carries its reason in the catalog.

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

`mojo` and `simdjson` install as transitive dependencies. The simdjson FFI wrapper builds on environment activation, no manual step. Nothing from Modular's MAX distribution is installed: the CPU path does not use it.

### GPU parsing is opt-in

The GPU pipeline needs `max-core`, which is governed by the [Modular Community License](https://www.modular.com/legal/community) rather than by this project's MIT licence. It is therefore not a dependency of `json`; you add it yourself:

```toml
[dependencies]
json     = { git = "https://github.com/ehsanmok/json.git", tag = "<latest-release>" }
max-core = ">=26.5.0"   # GPU only, Modular Community License
```

```mojo
from json.gpu import loads, load

var data = loads[target="gpu"](huge_json)
var file = load[target="gpu"]("huge.json")
```

Same `loads`, same `target` parameter as `json.loads`, and every non-GPU target is forwarded to the CPU parser, so switching backends is a change of import rather than a change of call. The import has to move because `json/parser.mojo` is the one module that must never reference the GPU code: Mojo resolves an import statement wherever it appears, even inside a `comptime if` branch that is false, so a single mention there would make `max-core` a requirement of `import json` for everyone.

Read the [Modular Community License](https://www.modular.com/legal/community) before you add that dependency: it places conditions on commercial and production use that MIT does not, and those are between you and Modular. See [`json/gpu/LICENSE-GPU.md`](./json/gpu/LICENSE-GPU.md).

Hardware: NVIDIA CUDA 7.0+, AMD ROCm 6+, or Apple Silicon. See [GPU compatibility](https://docs.modular.com/max/packages#gpu-compatibility).

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
| message | 16 KB | 15.6 us | 53.2 us | 12.4 us |
| document | 47 KB | 60.8 us | 186.0 us | 31.5 us |
| telemetry | 43 KB | 82.0 us | 141.6 us | 117.4 us |
| strings | 43 KB | 61.2 us | 109.6 us | 33.7 us |
| event | 27 KB | 38.8 us | 93.5 us | 14.7 us |

The middle column is what reading the same document costs if you route
it through `loads` and pick fields off `Value` by hand -- which is the
right tool when you are exploring a document, and the wrong one when
you already know its shape.

Telemetry is the slowest shape to write: 32 floats per record, and
shortest round-trip float formatting is the most expensive thing this
library does. It went from 166 to 117 us in 0.4.0 by dividing by
constants in the Grisu2 digit loop; see
[`docs/performance.md`](./docs/performance.md).

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

`json` is [MIT](./LICENSE). What it builds against is not all under the same terms, so the full picture:

| Component | Terms | Where to read them |
|---|---|---|
| `json` | MIT | [`LICENSE`](./LICENSE) |
| Mojo toolchain | The compiler and standard library sources are Apache-2.0 with LLVM Exceptions. The `mojo` and `mojo-compiler` conda packages you actually install still declare `LicenseRef-Modular-Proprietary` and ship the Modular Community License Terms. | `info/licenses/LICENSE` inside the installed package |
| simdjson | Apache-2.0. `libsimdjson_wrapper.so`, which this package builds and ships, links it. | [`NOTICE`](./NOTICE) |
| MAX (`max-core`) | **Modular Community License**, not an open-source licence. Needed only for the GPU path, never installed by depending on `json`. | [`json/gpu/LICENSE-GPU.md`](./json/gpu/LICENSE-GPU.md) |

The published package declares `mojo` and `simdjson` as dependencies and `max-core` only as a constraint, so installing `json` brings in nothing under the Modular Community License.
