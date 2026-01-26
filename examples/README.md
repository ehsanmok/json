# mojson Examples

This directory contains examples demonstrating the mojson library API.

## Running Examples

From the project root directory:

```bash
# Run all examples
pixi run examples

# Run individual examples
pixi run example-basic        # 01_basic_parsing.mojo
pixi run example-files        # 02_file_operations.mojo
pixi run example-value        # 03_value_types.mojo
pixi run example-gpu          # 04_gpu_parsing.mojo (requires GPU)
pixi run example-errors       # 05_error_handling.mojo
pixi run example-deserialize  # 06_deserialization.mojo

# Or run directly with mojo (from project root)
pixi run mojo -I . examples/01_basic_parsing.mojo
```

## Examples Overview

### 01_basic_parsing.mojo
Basic JSON parsing with `loads()` and `dumps()` for string-based operations.
- Parse JSON strings into `Value` objects
- Serialize `Value` objects back to JSON strings
- Handle different JSON types: int, float, string, bool, null, array, object
- Roundtrip parsing demonstration

### 02_file_operations.mojo
File I/O with `load()` and `dump()` for reading/writing JSON files.
- Read JSON from files with `load()`
- Write JSON to files with `dump()`
- Roundtrip file operations

### 03_value_types.mojo
Working with the `Value` type and its methods.
- Create `Value` objects directly (not from parsing)
- Type checking: `is_null()`, `is_bool()`, `is_int()`, `is_float()`, `is_string()`, `is_array()`, `is_object()`, `is_number()`
- Value extraction: `bool_value()`, `int_value()`, `float_value()`, `string_value()`
- Array/object metadata: `array_count()`, `object_count()`, `object_keys()`, `raw_json()`
- Value equality comparison

### 04_gpu_parsing.mojo
GPU-accelerated parsing for high-performance scenarios.
- Use `loads[target="gpu"]()` for GPU parsing
- Use `load[target="gpu"]()` for GPU file parsing
- Understanding when GPU parsing is beneficial (large documents)

### 05_error_handling.mojo
Handling JSON parse errors gracefully.
- Using try/except for error handling
- Batch processing with error recovery
- Examples of valid and invalid JSON

### 06_struct_serde.mojo
Type-safe struct serialization and deserialization.
- Implement `Serializable` trait with `to_json()` method
- Implement `Deserializable` trait with `from_json()` static method
- Use `serialize()` and `deserialize[T]()` helpers
- GPU-accelerated deserialization support
- Full round-trip examples

### 07_ndjson.mojo
NDJSON (Newline-Delimited JSON) parsing and streaming.
- Parse NDJSON strings with `loads[format="ndjson"]`
- Load .ndjson files with auto-detection
- GPU-accelerated NDJSON parsing
- Serialize to NDJSON format
- Stream large files with `load[streaming=True]`

### 08_lazy_parsing.mojo
On-demand lazy parsing for large documents.
- Create `LazyValue` with `loads[lazy=True]`
- Access specific paths without parsing entire document
- Type-specific getters: `get_string`, `get_int`, `get_bool`
- Chain lazy access with `[]` operator
- When to use lazy vs full parsing

### 09_jsonpath.mojo
JSONPath query language for extracting data.
- Basic path access: `$.store.name`
- Array indexing: `[0]`, `[-1]`, `[*]`
- Recursive descent: `$..name`
- Filter expressions: `[?@.price>30]`
- Array slices: `[0:2]`, `[::2]`

### 10_schema_validation.mojo
JSON Schema validation for data quality.
- Type validation: string, number, integer, object, array
- Required fields and property constraints
- Number constraints: minimum, maximum
- String constraints: minLength, maxLength
- Array constraints: items, minItems, maxItems
- Enum values and schema composition (allOf, anyOf)

### 11_json_patch.mojo
JSON Patch (RFC 6902) and Merge Patch (RFC 7396).
- Operations: add, remove, replace, move, copy, test
- Multiple operations in sequence
- Merge Patch for simpler partial updates
- Create diff patches between documents


## More Information

- [API Reference](../docs/api.md) — Complete function reference
- [Architecture](../docs/architecture.md) — CPU/GPU backend design
- [Performance](../docs/performance.md) — Optimization details
