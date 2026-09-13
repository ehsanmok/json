# Licensing of the GPU path

This directory is part of `json` and is licensed under the MIT License,
the same as the rest of the project. See `LICENSE` at the repository
root. Nothing from Modular is copied into, bundled with, or
redistributed by this project.

What is different here is the dependency. Every other part of `json`
builds against the Mojo standard library alone. This directory imports
`max.gpu`, `max.gpu.host`, `max.gpu.memory` and `max.gpu.primitives`,
which ship in `lib/mojo/max.mojoc` inside the `max-core` conda package.
Installing and using that package means accepting the **Modular
Community License**, which is not an open-source licence.

That is why this path is opt-in. Depending on `json` installs no MAX
component and compiles no file in this directory. The GPU entry points
are reached only by importing them explicitly:

```mojo
from json.gpu import loads_gpu
```

and that build needs `max-core` in the environment:

```toml
[dependencies]
json     = { git = "https://github.com/ehsanmok/json.git", tag = "v0.4.0" }
max-core = ">=26.5.0"   # GPU only, Modular Community License
```

## Which text governs your copy

The licence ships inside the package you install, at
`info/licenses/LICENSE` in the `max-core` artifact. On this machine
that resolves to a path like:

```
~/Library/Caches/rattler/cache/pkgs/max-core-<version>/info/licenses/LICENSE   # macOS
~/.cache/rattler/cache/pkgs/max-core-<version>/info/licenses/LICENSE           # Linux
```

Read that file rather than this summary. The current public text is at
<https://www.modular.com/legal/community>, and it is not always the same
as the one in a given release: the `max-core` 26.5.0 packages carry
"Modular Community License Terms, Last Modified April 12th, 2025",
while the version published on 18 August 2026 differs in at least one
respect that matters here (below).

## What the licence asks of you

Section 3 of the April 2025 terms sets out Distribution Requirements
for anything built on the SDK. Two bear directly on redistribution:

> (f) Licensee shall include all proprietary notices, labels, and marks
> provided by Modular in the Documentation (or elsewhere) in connection
> with the Redistributable Components in all copies of Applications
> that incorporate the Redistributable Components; and
>
> (g) The text of this Agreement shall be conspicuously displayed in
> each original or modified copy of the SDK.

This file exists to carry that notice forward to anyone who reaches the
GPU path through `json`. Because `json` redistributes no MAX component,
the obligations that attach to redistribution fall on whoever ships an
application built with it, not on this project.

## The accelerator limit

The April 2025 text restricts commercial use by device type:

> 2.2 Usage for production, commercial usage
>
> No capacity restrictions for any physical devices that are marketed
> as CPUs with X86 or ARM instruction set architectures, and/or
> NVIDIA-manufactured hardware products. However, for other device
> types not aforementioned, capacity restrictions are limited to the
> aggregate of no more than eight (8) other discrete physical
> accelerator devices in any computing environment regardless of type
> or configuration (the "Permitted Capacity").

This directory targets NVIDIA, AMD and Apple Metal. Under those terms
the AMD and Apple Metal targets fall under "other device types" and are
capped at eight accelerators for production or commercial use. The
August 2026 text removed that cap. Which one applies to you depends on
the version of `max-core` you installed, so check the file in your own
package before relying on either reading.

Nothing in this file is legal advice.

## Trademarks

"Mojo", "MAX" and "Modular" are used here only to name the software
this code depends on. Section 1 of the terms reserves those marks and
grants no licence to them.
