# json Examples

Worked examples are organised by progression so a new reader can walk
top-to-bottom and a returning reader can jump to the tier they need:

```
examples/
├── basic/         first ten lines of code -- start here
├── intermediate/  real-world feature usage
├── advanced/      perf-focused (GPU, lazy)
```

The examples on disk are the same files we run as part of `pixi run
tests`, so every snippet in here is known to compile and execute.

## Running examples

From the project root:

```bash
# Run every example in basic/ + intermediate/ + advanced/
pixi run examples

# Run a single example
pixi run example-parsing            # basic/parsing.mojo
pixi run example-files              # basic/file_io.mojo
pixi run example-value              # basic/value_types.mojo
pixi run example-errors             # basic/error_handling.mojo
pixi run example-reflection         # intermediate/reflection_serde.mojo
pixi run example-serde              # intermediate/struct_serde.mojo
pixi run example-ndjson             # intermediate/ndjson.mojo
pixi run example-jsonpath           # intermediate/jsonpath.mojo
pixi run example-schema             # intermediate/schema_validation.mojo
pixi run example-patch              # intermediate/json_patch.mojo
pixi run example-lazy               # advanced/lazy_parsing.mojo
pixi run example-gpu                # advanced/gpu_parsing.mojo

# Or invoke mojo directly
pixi run mojo -I . examples/basic/parsing.mojo
```

## basic/ -- first ten lines of code

Start here if you've never used `json` before. Each example fits on
one screen and exercises one entry point.

| File | What it covers |
|------|---|
| `basic/parsing.mojo`        | `loads()` / `dumps()` round trip, scalar & nested types |
| `basic/file_io.mojo`        | `load()` / `dump()` for reading and writing JSON files |
| `basic/value_types.mojo`    | The `Value` type: `is_*`, `*_value`, `array_count`, `object_keys` |
| `basic/error_handling.mojo` | `try`/`except` patterns, batch-process with error recovery |

Recommended order: `parsing` -> `value_types` -> `file_io` ->
`error_handling`.

## intermediate/ -- real-world feature usage

Once `loads`/`dumps` and `Value` feel natural, these examples introduce
the typed serde paths (reflection + manual traits), NDJSON, JSONPath,
JSON Schema, and JSON Patch. Pick whichever slice you need; they don't
depend on each other.

| File | What it covers |
|------|---|
| `intermediate/reflection_serde.mojo` | Zero-boilerplate struct serde via compile-time reflection (the path most apps want) |
| `intermediate/struct_serde.mojo`     | Manual `Serializable` / `Deserializable` traits when you need full control |
| `intermediate/ndjson.mojo`           | `loads[format="ndjson"]` and `load[format="ndjson"]` for line-delimited JSON |
| `intermediate/jsonpath.mojo`         | RFC 9535 JSONPath: `$..name`, filters, slices, recursive descent |
| `intermediate/schema_validation.mojo`| JSON Schema validation: `validate`, `is_valid`, type / range / enum constraints |
| `intermediate/json_patch.mojo`       | RFC 6902 JSON Patch + RFC 7396 Merge Patch (`apply_patch`, `merge_patch`) |

Recommended order: `reflection_serde` -> `struct_serde` -> the
RFC-specific ones in any order.

## advanced/ -- perf-focused

These exercise the parts of the library you only reach for once you
have a real workload to optimise.

| File | What it covers |
|------|---|
| `advanced/lazy_parsing.mojo` | `loads[lazy=True]` + `LazyValue`: type-specific getters, path-based access, when lazy beats full parse |
| `advanced/gpu_parsing.mojo`  | `loads[target="gpu"]` / `load[target="gpu"]` imported from `json.gpu`: GPU pipeline on NVIDIA, AMD, and Apple Metal. Needs `max-core`; see [`json/gpu/LICENSE-GPU.md`](../json/gpu/LICENSE-GPU.md) |

`advanced/gpu_parsing.mojo` runs the real GPU pipeline on NVIDIA,
AMD, and Apple Metal hosts. On hosts without any accelerator it
prints a guidance message and exits cleanly. On Xcode 26.x, run
`xcodebuild -downloadComponent MetalToolchain` once if `mojo build`
reports `Metal Compiler failed to compile metallib`.

## Discovering more

- [`docs/api.md`](../docs/api.md) -- complete function reference.
- [`docs/architecture.md`](../docs/architecture.md) -- how the
  parser, `Document` / `Tape`, and the CPU / GPU backends fit
  together.
- [`docs/performance.md`](../docs/performance.md) -- benchmark
  numbers and tuning notes.
