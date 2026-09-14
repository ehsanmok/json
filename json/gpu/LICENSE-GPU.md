# Licensing of the GPU path

This directory is part of `json` and is licensed under the MIT License,
the same as the rest of the project. See `LICENSE` at the repository
root. Nothing from Modular is copied into, bundled with, or
redistributed by this project.

**Using it requires `max-core`, which is governed by the [Modular
Community License](https://www.modular.com/legal/community). That is
not an open-source licence.** Every other part of `json` builds against
the Mojo standard library alone; this directory imports `max.gpu`,
`max.gpu.host`, `max.gpu.memory` and `max.gpu.primitives`, which ship
in `max-core`.

That is why the GPU path is opt-in. Depending on `json` installs no MAX
component and compiles no file in this directory. Reaching it takes an
import from this module and a dependency you add yourself:

```mojo
from json.gpu import loads, load

var data = loads[target="gpu"](huge_json)
var file = load[target="gpu"]("huge.json")
```

```toml
[dependencies]
json     = { git = "https://github.com/ehsanmok/json.git", tag = "v0.4.0" }
max-core = ">=26.5.0"   # GPU only, Modular Community License
```

`json.gpu.loads` takes the same `target` parameter as `json.loads` and
forwards every non-GPU target to it, so switching backends is a change
of import rather than a change of call.

Read the Modular Community License before you add that line. It places
conditions on commercial and production use that the MIT licence on
this code does not, and those conditions are between you and Modular.

"Mojo", "MAX" and "Modular" are used here only to name the software
this code depends on. The licence reserves those marks and grants no
rights to them.

Nothing in this file is legal advice.
