# GPU JSON Parsing Module
#
# Opt-in. Nothing on the CPU path imports this package, which is what
# keeps `import json` free of any MAX dependency; see `loads.mojo` for
# why that is a structural property rather than a convention.
#
# Licensing: the code here is MIT like the rest of the library, but
# using it requires `max-core`, which is governed by the Modular
# Community License. See `LICENSE-GPU.md` in this directory.
#
# - loads.mojo: entry points (loads_gpu, load_gpu, loads_ndjson_gpu)
# - parser.mojo: GPU parsing pipeline (parse_json_gpu, parse_json_gpu_from_pinned)
# - kernels.mojo: GPU kernel implementations
# - stream_compact.mojo: GPU stream compaction for position extraction
# - tape_adapter.mojo: structural positions -> stage 2 -> Document

from .loads import loads_gpu, load_gpu, loads_ndjson_gpu
from .parser import parse_json_gpu, parse_json_gpu_from_pinned
from .kernels import BLOCK_SIZE_OPT, fused_json_kernel
from .stream_compact import extract_positions_gpu_lean
from .tape_adapter import parse_gpu_to_value
