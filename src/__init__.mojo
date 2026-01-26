# High-performance JSON library for Mojo
#
# Core API: loads, dumps, load, dump
#
# Usage:
#   from mojson import loads, dumps, load, dump, Value
#
#   # Parse JSON
#   var data = loads('{"name": "Alice"}')
#   var data = loads[target="gpu"](large_json)     # GPU for large files
#   var data = loads(json, config)                 # With parser config
#   var values = loads[format="ndjson"](ndjson)    # NDJSON -> List[Value]
#   var lazy = loads[lazy=True](huge_json)         # Lazy parsing
#
#   # Serialize JSON
#   var s = dumps(data)                            # Compact
#   var s = dumps(data, indent="  ")               # Pretty
#   var s = dumps[format="ndjson"](values)         # List[Value] -> NDJSON
#
#   # File operations
#   with open("data.json", "r") as f:
#       var data = load(f)
#   var parser = load[streaming=True]("big.ndjson")  # Streaming
#
#   with open("out.json", "w") as f:
#       dump(data, f)

# Core API
from .value import Value, Null
from .parser import loads, load
from .serialize import dumps, dump
from .config import ParserConfig, SerializerConfig

# Value helpers
from .value import make_array_value, make_object_value

# Struct serialization
from .serialize import to_json_value, to_json_string, Serializable, serialize
from .deserialize import (
    get_string, get_int, get_bool, get_float,
    Deserializable, deserialize,
)

# Advanced features
from .patch import apply_patch, merge_patch, create_merge_patch
from .jsonpath import jsonpath_query, jsonpath_one
from .schema import validate, is_valid, ValidationResult, ValidationError

# Internal types (for advanced use)
from .lazy import LazyValue
from .streaming import StreamingParser, ArrayStreamingParser
