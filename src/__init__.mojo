# mojson - High-performance JSON library for Mojo
# Python-compatible API: loads, dumps, load, dump
#
# Usage:
#   from src import loads, dumps, load, dump, Value
#
#   # String operations
#   var data = loads('{"key": "value"}')           # CPU (default)
#   var data = loads[target="gpu"]('{"key": 1}')   # GPU
#   print(dumps(data))
#
#   # File operations
#   with open("data.json", "r") as f:
#       var data = load(f)                         # CPU (default)
#       var data = load[target="gpu"](f)           # GPU
#
#   with open("output.json", "w") as f:
#       dump(data, f)

from .value import Value, Null, make_array_value, make_object_value
from .parser import loads, load
from .serialize import dumps, dump
