# GPU backend for the `json` library.
#
# Opt-in. Nothing on the CPU path imports this package, which is what
# keeps `import json` free of any MAX dependency; see `backend.mojo`
# for why that is a structural property rather than a convention.
#
# Licensing: the code here is MIT like the rest of the library, but
# using it requires `max-core`, which is governed by the Modular
# Community License. See `LICENSE-GPU.md` in this directory.
#
# - backend.mojo: `loads` / `load`, the same names and the same
#   `target` parameter as in `json`, running on the GPU by default
# - parser.mojo: GPU parsing pipeline (parse_json_gpu, parse_json_gpu_from_pinned)
# - kernels.mojo: GPU kernel implementations
# - stream_compact.mojo: GPU stream compaction for position extraction
# - tape_adapter.mojo: structural positions -> stage 2 -> Document

from .backend import loads, load
from .parser import parse_json_gpu, parse_json_gpu_from_pinned
from .kernels import BLOCK_SIZE_OPT, fused_json_kernel
from .stream_compact import extract_positions_gpu_lean
from .tape_adapter import parse_gpu_to_value
