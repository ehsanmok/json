# mojson CPU backends
# Supports multiple backends:
#   - simdjson (FFI): High-performance C++ backend (default)
#   - mojo (native): Pure Mojo implementation (zero FFI)

# =============================================================================
# Common Types (backend-agnostic)
# =============================================================================

from .types import (
    JSON_TYPE_NULL,
    JSON_TYPE_BOOL,
    JSON_TYPE_INT64,
    JSON_TYPE_UINT64,
    JSON_TYPE_DOUBLE,
    JSON_TYPE_STRING,
    JSON_TYPE_ARRAY,
    JSON_TYPE_OBJECT,
    JSON_OK,
    JSON_ERROR_INVALID,
    JSON_ERROR_CAPACITY,
    JSON_ERROR_UTF8,
    JSON_ERROR_OTHER,
)

# =============================================================================
# simdjson Backend (FFI)
# =============================================================================

from .simdjson_ffi import (
    SimdjsonFFI,
    SIMDJSON_OK,
    SIMDJSON_ERROR_INVALID_JSON,
    SIMDJSON_ERROR_CAPACITY,
    SIMDJSON_ERROR_UTF8,
    SIMDJSON_ERROR_OTHER,
    SIMDJSON_TYPE_NULL,
    SIMDJSON_TYPE_BOOL,
    SIMDJSON_TYPE_INT64,
    SIMDJSON_TYPE_UINT64,
    SIMDJSON_TYPE_DOUBLE,
    SIMDJSON_TYPE_STRING,
    SIMDJSON_TYPE_ARRAY,
    SIMDJSON_TYPE_OBJECT,
)

# =============================================================================
# Mojo Backend (Pure Native)
# =============================================================================

from .mojo_backend import parse_mojo, MojoJSONParser

# =============================================================================
# SIMD Backend (High-Performance Native)
# =============================================================================

from .simd_backend import parse_simd, FastParser
