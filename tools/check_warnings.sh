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

# The package on its own, which is what a consumer compiles against.
mojo doc json -o "$OUT/doc.json" > "$OUT/doc.log" 2>&1
report "mojo doc json" "$OUT/doc.log"

targets=(tests/*.mojo benchmark/mojo/*.mojo examples/*/*.mojo)
for f in "${targets[@]}"; do
  name=$(basename "$f" .mojo)
  mojo build -I . -o "$OUT/$name" "$f" > "$OUT/$name.log" 2>&1
  report "$f" "$OUT/$name.log"
done

if [[ "$status" == "0" ]]; then
  echo "No warnings in $((${#targets[@]} + 1)) targets."
fi
exit "$status"
