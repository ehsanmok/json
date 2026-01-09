# mojson Examples

This directory contains examples demonstrating the mojson library API.

## Running Examples

From the project root directory:

```bash
# Run all examples
pixi run examples

# Run individual examples
pixi run example-basic   # 01_basic_parsing.mojo
pixi run example-files   # 02_file_operations.mojo
pixi run example-value   # 03_value_types.mojo
pixi run example-gpu     # 04_gpu_parsing.mojo (requires GPU)
pixi run example-errors  # 05_error_handling.mojo

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

## API Quick Reference

| Function | Description | Example |
|----------|-------------|---------|
| `loads(s)` | Parse JSON string (CPU) | `var v = loads('{"a":1}')` |
| `loads[target="gpu"](s)` | Parse JSON string (GPU) | `var v = loads[target="gpu"]('{"a":1}')` |
| `dumps(v)` | Serialize to string | `var s = dumps(v)` |
| `load(f)` | Parse from file (CPU) | `var v = load(f)` |
| `load[target="gpu"](f)` | Parse from file (GPU) | `var v = load[target="gpu"](f)` |
| `dump(v, f)` | Write to file | `dump(v, f)` |
