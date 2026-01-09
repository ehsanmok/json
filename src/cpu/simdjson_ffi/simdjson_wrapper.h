// simdjson C wrapper for Mojo FFI
// Provides a simple C interface to simdjson's high-performance JSON parsing

#ifndef SIMDJSON_WRAPPER_H
#define SIMDJSON_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Opaque handles
typedef void* simdjson_parser_t;
typedef void* simdjson_value_t;
typedef void* simdjson_array_iter_t;
typedef void* simdjson_object_iter_t;

// Result codes
#define SIMDJSON_OK 0
#define SIMDJSON_ERROR_INVALID_JSON 1
#define SIMDJSON_ERROR_CAPACITY 2
#define SIMDJSON_ERROR_UTF8 3
#define SIMDJSON_ERROR_OTHER 99

// Value types
#define SIMDJSON_TYPE_NULL 0
#define SIMDJSON_TYPE_BOOL 1
#define SIMDJSON_TYPE_INT64 2
#define SIMDJSON_TYPE_UINT64 3
#define SIMDJSON_TYPE_DOUBLE 4
#define SIMDJSON_TYPE_STRING 5
#define SIMDJSON_TYPE_ARRAY 6
#define SIMDJSON_TYPE_OBJECT 7

// Parser lifecycle
simdjson_parser_t simdjson_create_parser(void);
void simdjson_destroy_parser(simdjson_parser_t parser);

// Parse JSON string - returns root element handle
int simdjson_parse(simdjson_parser_t parser, const char* json, size_t len);

// Get root element
simdjson_value_t simdjson_get_root(simdjson_parser_t parser);

// Value type inspection
int simdjson_value_get_type(simdjson_value_t value);

// Scalar value access
int simdjson_value_get_bool(simdjson_value_t value, int* out);
int simdjson_value_get_int64(simdjson_value_t value, int64_t* out);
int simdjson_value_get_uint64(simdjson_value_t value, uint64_t* out);
int simdjson_value_get_double(simdjson_value_t value, double* out);
int simdjson_value_get_string(simdjson_value_t value, const char** data, size_t* len);

// Array iteration
simdjson_array_iter_t simdjson_array_begin(simdjson_value_t value);
int simdjson_array_iter_done(simdjson_array_iter_t iter);
simdjson_value_t simdjson_array_iter_get(simdjson_array_iter_t iter);
void simdjson_array_iter_next(simdjson_array_iter_t iter);
void simdjson_array_iter_free(simdjson_array_iter_t iter);
size_t simdjson_array_count(simdjson_value_t value);

// Object iteration
simdjson_object_iter_t simdjson_object_begin(simdjson_value_t value);
int simdjson_object_iter_done(simdjson_object_iter_t iter);
void simdjson_object_iter_get_key(simdjson_object_iter_t iter, const char** data, size_t* len);
simdjson_value_t simdjson_object_iter_get_value(simdjson_object_iter_t iter);
void simdjson_object_iter_next(simdjson_object_iter_t iter);
void simdjson_object_iter_free(simdjson_object_iter_t iter);
size_t simdjson_object_count(simdjson_value_t value);

// Value handle cleanup (for values from iterators)
void simdjson_value_free(simdjson_value_t value);

// Required padding constant
size_t simdjson_required_padding(void);

#ifdef __cplusplus
}
#endif

#endif // SIMDJSON_WRAPPER_H
