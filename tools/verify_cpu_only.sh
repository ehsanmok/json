#!/usr/bin/env bash
# tools/verify_cpu_only.sh -- assert the default environment has no MAX.
#
# The default pixi environment is meant to look exactly like what a
# consumer of the published conda package gets: Mojo and simdjson, and
# nothing from Modular's MAX distribution. If `max-core` reappears here
# it means some module on the CPU import graph has started naming
# `json.gpu` again, which puts a proprietary-licensed dependency back
# into every downstream lock file. That is the regression this check
# exists to catch, so it fails the build rather than warning.
set -euo pipefail

names=$(pixi list --json | python3 -c '
import json, sys
print(" ".join(sorted(p["name"] for p in json.load(sys.stdin))))
')

bad=""
for pkg in max max-core; do
  case " $names " in
    *" $pkg "*) bad="$bad $pkg" ;;
  esac
done

if [ -n "$bad" ]; then
  echo "FAIL: the default environment contains:$bad"
  echo "      Something on the CPU import graph is importing json.gpu."
  exit 1
fi

echo "OK: the default environment has no MAX packages."
