// simdjson C wrapper implementation for Mojo FFI
#include "simdjson_wrapper.h"

// <iterator> must precede <simdjson.h>. simdjson 4.6.11 uses
// std::inserter in its schema-intersection helpers without including
// <iterator> itself. libstdc++ happens to pull it in transitively, so
// Linux builds fine; libc++ does not, and the conda macOS toolchain
// fails the package build with "no member named 'inserter' in
// namespace 'std'". Including it here keeps us independent of whichever
// simdjson the solver picks.
#include <iterator>

#include <simdjson.h>

#include <pthread.h>

#include <cstring>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

// Internal parser state
struct SimdjsonParserState {
    simdjson::dom::parser parser;
    simdjson::dom::element root_element;
    simdjson::padded_string current_padded;
    bool has_element = false;
    std::vector<uint64_t> tape;
    std::vector<uint64_t> scratch;
    std::string arena;
    std::vector<double> floats;
    std::vector<int64_t> ints;
};

namespace {

// Mirrors json/document.mojo: a 4-bit tag over a 60-bit payload, a pair is
// two 30-bit halves, and a container's children precede its header.
constexpr uint64_t TAG_NULL = 0, TAG_BOOL = 1, TAG_INT = 2, TAG_FLOAT = 3,
                   TAG_STRING = 4, TAG_ARRAY = 5, TAG_OBJECT = 6,
                   TAG_KEY_INLINE = 9, TAG_INT_POOL = 10, TAG_UINT = 11;
constexpr uint64_t PAYLOAD_MASK = (uint64_t(1) << 60) - 1;
constexpr uint64_t HALF_MASK = (uint64_t(1) << 30) - 1;
constexpr int64_t INLINE_INT_LIMIT = int64_t(1) << 59;

inline uint64_t entry(uint64_t tag, uint64_t payload) {
    return (tag << 60) | (payload & PAYLOAD_MASK);
}

inline uint64_t pair(uint64_t hi, uint64_t lo) {
    return ((hi & HALF_MASK) << 30) | (lo & HALF_MASK);
}

struct TapeTooLarge {};

uint64_t slice(SimdjsonParserState& st, std::string_view sv, uint64_t tag) {
    uint64_t off = st.arena.size();
    if (off > HALF_MASK || sv.size() > HALF_MASK ||
        off + sv.size() > HALF_MASK)
        throw TapeTooLarge{};
    st.arena.append(sv.data(), sv.size());
    return entry(tag, pair(off, sv.size()));
}

uint64_t emit(SimdjsonParserState& st, simdjson::dom::element e) {
    using simdjson::dom::element_type;
    switch (e.type()) {
    case element_type::NULL_VALUE:
        return entry(TAG_NULL, 0);
    case element_type::BOOL:
        return entry(TAG_BOOL, bool(e.get_bool().value_unsafe()) ? 1 : 0);
    case element_type::INT64: {
        int64_t v = e.get_int64().value_unsafe();
        if (v >= -INLINE_INT_LIMIT && v < INLINE_INT_LIMIT)
            return entry(TAG_INT, uint64_t(v));
        st.ints.push_back(v);
        return entry(TAG_INT_POOL, st.ints.size() - 1);
    }
    case element_type::UINT64: {
        uint64_t v = e.get_uint64().value_unsafe();
        int64_t bits;
        std::memcpy(&bits, &v, sizeof bits);
        st.ints.push_back(bits);
        return entry(TAG_UINT, st.ints.size() - 1);
    }
    case element_type::DOUBLE:
        st.floats.push_back(e.get_double().value_unsafe());
        return entry(TAG_FLOAT, st.floats.size() - 1);
    case element_type::STRING:
        return slice(st, e.get_string().value_unsafe(), TAG_STRING);
    case element_type::ARRAY: {
        size_t base = st.scratch.size();
        uint64_t count = 0;
        simdjson::dom::array arr = e.get_array().value_unsafe();
        for (auto child : arr) {
            uint64_t h = emit(st, child);
            st.scratch.push_back(h);
            ++count;
        }
        uint64_t start = st.tape.size();
        st.tape.insert(st.tape.end(), st.scratch.begin() + base, st.scratch.end());
        st.scratch.resize(base);
        if (count > HALF_MASK || start > HALF_MASK) throw TapeTooLarge{};
        return entry(TAG_ARRAY, pair(count, start));
    }
    case element_type::OBJECT: {
        size_t base = st.scratch.size();
        uint64_t count = 0;
        simdjson::dom::object obj = e.get_object().value_unsafe();
        for (auto field : obj) {
            uint64_t k = slice(st, field.key, TAG_KEY_INLINE);
            uint64_t v = emit(st, field.value);
            st.scratch.push_back(k);
            st.scratch.push_back(v);
            ++count;
        }
        uint64_t start = st.tape.size();
        st.tape.insert(st.tape.end(), st.scratch.begin() + base, st.scratch.end());
        st.scratch.resize(base);
        if (count > HALF_MASK || start > HALF_MASK) throw TapeTooLarge{};
        return entry(TAG_OBJECT, pair(count, start));
    }
    }
    return entry(TAG_NULL, 0);
}

}  // namespace

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

// A pthread key rather than thread_local: this library is loaded with
// dlopen, and a TLS segment in it would bump glibc's TLS generation and
// leave every other library's dynamic TLS access (the host's allocator
// included) on __tls_get_addr's slow path.
static pthread_key_t thread_parser_key;
static pthread_once_t thread_parser_once = PTHREAD_ONCE_INIT;

static void thread_parser_key_init() {
    pthread_key_create(&thread_parser_key, [](void* p) {
        delete static_cast<SimdjsonParserState*>(p);
    });
}

simdjson_parser_t simdjson_thread_parser(void) {
    pthread_once(&thread_parser_once, thread_parser_key_init);
    auto* st = static_cast<SimdjsonParserState*>(
        pthread_getspecific(thread_parser_key));
    if (!st) {
        st = new SimdjsonParserState();
        pthread_setspecific(thread_parser_key, st);
    }
    return st;
}

int simdjson_parse_to_tape(simdjson_parser_t parser, const char* json, size_t len) {
    if (!parser || !json) return SIMDJSON_ERROR_OTHER;
    auto* st = static_cast<SimdjsonParserState*>(parser);
    st->has_element = false;
    auto result = st->parser.parse(json, len, true);
    if (result.error()) {
        auto ec = result.error();
        if (ec == simdjson::CAPACITY) return SIMDJSON_ERROR_CAPACITY;
        if (ec == simdjson::UTF8_ERROR) return SIMDJSON_ERROR_UTF8;
        return SIMDJSON_ERROR_INVALID_JSON;
    }
    st->root_element = result.value();
    st->has_element = true;
    st->tape.clear();
    st->scratch.clear();
    st->arena.clear();
    st->floats.clear();
    st->ints.clear();
    try {
        uint64_t root = emit(*st, st->root_element);
        st->tape.push_back(root);
    } catch (const TapeTooLarge&) {
        return SIMDJSON_ERROR_CAPACITY;
    }
    return SIMDJSON_OK;
}

void simdjson_tape_buffers(simdjson_parser_t parser, int64_t* out) {
    auto* st = static_cast<SimdjsonParserState*>(parser);
    out[0] = int64_t(reinterpret_cast<intptr_t>(st->tape.data()));
    out[1] = int64_t(st->tape.size());
    out[2] = int64_t(reinterpret_cast<intptr_t>(st->arena.data()));
    out[3] = int64_t(st->arena.size());
    out[4] = int64_t(reinterpret_cast<intptr_t>(st->floats.data()));
    out[5] = int64_t(st->floats.size());
    out[6] = int64_t(reinterpret_cast<intptr_t>(st->ints.data()));
    out[7] = int64_t(st->ints.size());
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

void simdjson_memcpy_from_addr(void* dst, intptr_t src_addr, size_t n) {
    memcpy(dst, reinterpret_cast<const void*>(src_addr), n);
}

} // extern "C"
