#!/usr/bin/env bash
# Build everything and fail on any compiler warning.
#
# A warning here is usually a deprecation, and a library's deprecation
# warnings surface in every downstream consumer's build -- they are
# reported when the consumer instantiates the code, not when we compile
# it. So "no warnings" is a property of the published artifact, not a
# tidiness preference, and it needs a gate rather than a habit.
#
# Usage: pixi run -e dev warnings-check
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${TMPDIR:-/tmp}/json-warncheck"
mkdir -p "$OUT"

status=0
report() {
  local label="$1" log="$2"
  local warnings errors
  warnings=$(grep -c "warning:" "$log" || true)
  errors=$(grep -c "error:" "$log" || true)
  if [[ "$errors" != "0" ]]; then
    echo "FAIL (build) $label"
    grep "error:" "$log" | head -5 | sed 's/^/    /'
    status=1
  elif [[ "$warnings" != "0" ]]; then
    echo "FAIL (warnings) $label"
    grep "warning:" "$log" | sort -u | head -10 | sed 's/^/    /'
    status=1
  fi
}

# Each import root on its own, which is what a consumer compiles
# against. `json.gpu` is the opt-in GPU backend; it needs max-core, so
# this script only runs in an environment that has the gpu feature.
for pkg in json; do
  mojo doc "$pkg" -o "$OUT/doc-$pkg.json" > "$OUT/doc-$pkg.log" 2>&1
  report "mojo doc $pkg" "$OUT/doc-$pkg.log"
done

# Sources that import `json.gpu`. Compiling these needs a real
# accelerator, not just `max-core`: the kernels are instantiated for a
# concrete architecture, and on a machine without one the compiler
# stops with "Unknown GPU architecture detected." A GPU-less CI runner
# therefore has to skip them rather than fail.
gpu_targets=(
  tests/test_gpu.mojo
  tests/test_gpu_kernels.mojo
  tests/test_bracket_match.mojo
  tests/bench_bracket_match.mojo
  examples/advanced/gpu_parsing.mojo
  benchmark/mojo/bench_gpu.mojo
  benchmark/mojo/bench_gpu_apple.mojo
)

probe="$OUT/accel_probe.mojo"
cat > "$probe" <<'PROBE'
from std.sys import has_accelerator


def main() raises:
    print(has_accelerator())
PROBE
have_gpu=$(mojo run "$probe" 2>/dev/null | tail -1)

is_gpu_target() {
  local candidate="$1"
  for g in "${gpu_targets[@]}"; do
    [[ "$candidate" == "$g" ]] && return 0
  done
  return 1
}

targets=(tests/*.mojo benchmark/mojo/*.mojo examples/*/*.mojo)
built=0
skipped=0
for f in "${targets[@]}"; do
  if [[ "$have_gpu" != "True" ]] && is_gpu_target "$f"; then
    skipped=$((skipped + 1))
    continue
  fi
  name=$(basename "$f" .mojo)
  mojo build -I . -o "$OUT/$name" "$f" > "$OUT/$name.log" 2>&1
  report "$f" "$OUT/$name.log"
  built=$((built + 1))
done

if [[ "$status" == "0" ]]; then
  if [[ "$skipped" -gt 0 ]]; then
    echo "No warnings in $((built + 1)) targets ($skipped GPU targets skipped: no accelerator)."
  else
    echo "No warnings in $((built + 1)) targets."
  fi
fi
exit "$status"
