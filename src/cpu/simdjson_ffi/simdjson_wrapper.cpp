// simdjson C wrapper implementation for Mojo FFI
#include "simdjson_wrapper.h"
#include <simdjson.h>

#include <cstring>
#include <cstdio>
#include <string>

// Internal parser state
struct SimdjsonParserState {
    simdjson::dom::parser parser;
    simdjson::dom::element root_element;
    simdjson::padded_string current_padded;
    bool has_element;
};

// Wrapper for element (holds a copy)
struct SimdjsonValueWrapper {
    simdjson::dom::element element;
    bool owned;  // If true, this was allocated and needs cleanup
};

// Array iterator
struct SimdjsonArrayIter {
    simdjson::dom::array array;
    simdjson::dom::array::iterator current;
    simdjson::dom::array::iterator end;
};

// Object iterator
struct SimdjsonObjectIter {
    simdjson::dom::object object;
    simdjson::dom::object::iterator current;
    simdjson::dom::object::iterator end;
};

extern "C" {

simdjson_parser_t simdjson_create_parser(void) {
    auto* state = new SimdjsonParserState();
    state->has_element = false;
    return state;
}

void simdjson_destroy_parser(simdjson_parser_t parser) {
    if (parser) {
        delete static_cast<SimdjsonParserState*>(parser);
    }
}

int simdjson_parse(simdjson_parser_t parser, const char* json, size_t len) {
    if (!parser || !json) return SIMDJSON_ERROR_OTHER;

    auto* state = static_cast<SimdjsonParserState*>(parser);

    // Create padded string (copies the data)
    state->current_padded = simdjson::padded_string(json, len);

    auto result = state->parser.parse(state->current_padded);
    if (result.error()) {
        state->has_element = false;
        auto ec = result.error();
        if (ec == simdjson::CAPACITY) return SIMDJSON_ERROR_CAPACITY;
        if (ec == simdjson::UTF8_ERROR) return SIMDJSON_ERROR_UTF8;
        return SIMDJSON_ERROR_INVALID_JSON;
    }

    state->root_element = result.value();
    state->has_element = true;
    return SIMDJSON_OK;
}

simdjson_value_t simdjson_get_root(simdjson_parser_t parser) {
    if (!parser) return nullptr;
    auto* state = static_cast<SimdjsonParserState*>(parser);
    if (!state->has_element) return nullptr;

    auto* wrapper = new SimdjsonValueWrapper();
    wrapper->element = state->root_element;
    wrapper->owned = true;
    return wrapper;
}

int simdjson_value_get_type(simdjson_value_t value) {
    if (!value) return SIMDJSON_TYPE_NULL;
    auto* wrapper = static_cast<SimdjsonValueWrapper*>(value);

    switch (wrapper->element.type()) {
        case simdjson::dom::element_type::NULL_VALUE: return SIMDJSON_TYPE_NULL;
        case simdjson::dom::element_type::BOOL: return SIMDJSON_TYPE_BOOL;
        case simdjson::dom::element_type::INT64: return SIMDJSON_TYPE_INT64;
        case simdjson::dom::element_type::UINT64: return SIMDJSON_TYPE_UINT64;
        case simdjson::dom::element_type::DOUBLE: return SIMDJSON_TYPE_DOUBLE;
        case simdjson::dom::element_type::STRING: return SIMDJSON_TYPE_STRING;
        case simdjson::dom::element_type::ARRAY: return SIMDJSON_TYPE_ARRAY;
        case simdjson::dom::element_type::OBJECT: return SIMDJSON_TYPE_OBJECT;
        default: return SIMDJSON_TYPE_NULL;
    }
}

int simdjson_value_get_bool(simdjson_value_t value, int* out) {
    if (!value || !out) return SIMDJSON_ERROR_OTHER;
    auto* wrapper = static_cast<SimdjsonValueWrapper*>(value);

    auto result = wrapper->element.get_bool();
    if (result.error()) return SIMDJSON_ERROR_INVALID_JSON;
    *out = result.value() ? 1 : 0;
    return SIMDJSON_OK;
}

int simdjson_value_get_int64(simdjson_value_t value, int64_t* out) {
    if (!value || !out) return SIMDJSON_ERROR_OTHER;
    auto* wrapper = static_cast<SimdjsonValueWrapper*>(value);

    auto result = wrapper->element.get_int64();
    if (result.error()) return SIMDJSON_ERROR_INVALID_JSON;
    *out = result.value();
    return SIMDJSON_OK;
}

int simdjson_value_get_uint64(simdjson_value_t value, uint64_t* out) {
    if (!value || !out) return SIMDJSON_ERROR_OTHER;
    auto* wrapper = static_cast<SimdjsonValueWrapper*>(value);

    auto result = wrapper->element.get_uint64();
    if (result.error()) return SIMDJSON_ERROR_INVALID_JSON;
    *out = result.value();
    return SIMDJSON_OK;
}

int simdjson_value_get_double(simdjson_value_t value, double* out) {
    if (!value || !out) return SIMDJSON_ERROR_OTHER;
    auto* wrapper = static_cast<SimdjsonValueWrapper*>(value);

    auto result = wrapper->element.get_double();
    if (result.error()) return SIMDJSON_ERROR_INVALID_JSON;
    *out = result.value();
    return SIMDJSON_OK;
}

int simdjson_value_get_string(simdjson_value_t value, const char** data, size_t* len) {
    if (!value || !data || !len) return SIMDJSON_ERROR_OTHER;
    auto* wrapper = static_cast<SimdjsonValueWrapper*>(value);

    auto result = wrapper->element.get_string();
    if (result.error()) return SIMDJSON_ERROR_INVALID_JSON;
    auto sv = result.value();
    *data = sv.data();
    *len = sv.size();
    return SIMDJSON_OK;
}

// Array iteration
simdjson_array_iter_t simdjson_array_begin(simdjson_value_t value) {
    if (!value) return nullptr;
    auto* wrapper = static_cast<SimdjsonValueWrapper*>(value);

    auto arr_result = wrapper->element.get_array();
    if (arr_result.error()) return nullptr;

    auto* iter = new SimdjsonArrayIter();
    iter->array = arr_result.value();
    iter->current = iter->array.begin();
    iter->end = iter->array.end();
    return iter;
}

int simdjson_array_iter_done(simdjson_array_iter_t iter) {
    if (!iter) return 1;
    auto* it = static_cast<SimdjsonArrayIter*>(iter);
    return (it->current == it->end) ? 1 : 0;
}

simdjson_value_t simdjson_array_iter_get(simdjson_array_iter_t iter) {
    if (!iter) return nullptr;
    auto* it = static_cast<SimdjsonArrayIter*>(iter);
    if (it->current == it->end) return nullptr;

    auto* wrapper = new SimdjsonValueWrapper();
    wrapper->element = *it->current;
    wrapper->owned = true;
    return wrapper;
}

void simdjson_array_iter_next(simdjson_array_iter_t iter) {
    if (!iter) return;
    auto* it = static_cast<SimdjsonArrayIter*>(iter);
    if (it->current != it->end) {
        ++it->current;
    }
}

void simdjson_array_iter_free(simdjson_array_iter_t iter) {
    if (iter) {
        delete static_cast<SimdjsonArrayIter*>(iter);
    }
}

size_t simdjson_array_count(simdjson_value_t value) {
    if (!value) return 0;
    auto* wrapper = static_cast<SimdjsonValueWrapper*>(value);
    auto arr_result = wrapper->element.get_array();
    if (arr_result.error()) return 0;
    return arr_result.value().size();
}

// Object iteration
simdjson_object_iter_t simdjson_object_begin(simdjson_value_t value) {
    if (!value) return nullptr;
    auto* wrapper = static_cast<SimdjsonValueWrapper*>(value);

    auto obj_result = wrapper->element.get_object();
    if (obj_result.error()) return nullptr;

    auto* iter = new SimdjsonObjectIter();
    iter->object = obj_result.value();
    iter->current = iter->object.begin();
    iter->end = iter->object.end();
    return iter;
}

int simdjson_object_iter_done(simdjson_object_iter_t iter) {
    if (!iter) return 1;
    auto* it = static_cast<SimdjsonObjectIter*>(iter);
    return (it->current == it->end) ? 1 : 0;
}

void simdjson_object_iter_get_key(simdjson_object_iter_t iter, const char** data, size_t* len) {
    if (!iter || !data || !len) return;
    auto* it = static_cast<SimdjsonObjectIter*>(iter);
    if (it->current == it->end) {
        *data = nullptr;
        *len = 0;
        return;
    }
    auto key = it->current.key();
    *data = key.data();
    *len = key.size();
}

simdjson_value_t simdjson_object_iter_get_value(simdjson_object_iter_t iter) {
    if (!iter) return nullptr;
    auto* it = static_cast<SimdjsonObjectIter*>(iter);
    if (it->current == it->end) return nullptr;

    auto* wrapper = new SimdjsonValueWrapper();
    wrapper->element = it->current.value();
    wrapper->owned = true;
    return wrapper;
}

void simdjson_object_iter_next(simdjson_object_iter_t iter) {
    if (!iter) return;
    auto* it = static_cast<SimdjsonObjectIter*>(iter);
    if (it->current != it->end) {
        ++it->current;
    }
}

void simdjson_object_iter_free(simdjson_object_iter_t iter) {
    if (iter) {
        delete static_cast<SimdjsonObjectIter*>(iter);
    }
}

size_t simdjson_object_count(simdjson_value_t value) {
    if (!value) return 0;
    auto* wrapper = static_cast<SimdjsonValueWrapper*>(value);
    auto obj_result = wrapper->element.get_object();
    if (obj_result.error()) return 0;
    return obj_result.value().size();
}

void simdjson_value_free(simdjson_value_t value) {
    if (value) {
        auto* wrapper = static_cast<SimdjsonValueWrapper*>(value);
        if (wrapper->owned) {
            delete wrapper;
        }
    }
}

size_t simdjson_required_padding(void) {
    return simdjson::SIMDJSON_PADDING;
}

} // extern "C"
