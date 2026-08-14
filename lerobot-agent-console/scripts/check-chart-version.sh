#!/usr/bin/env bash
# Fail if a chart changed without its version being bumped.
#
# Why this exists: charts are published to an OCI registry by version tag. Pushing a modified
# chart under a version that already exists leaves two different charts answering to one number,
# and `helm pull --version X` then returns whichever won the race. Nothing in helm prevents that
# — `helm push` overwrites happily.
#
# This is the one check from helm/chart-testing (`ct lint --check-version-increment`) that we
# actually need. ct itself is a Go binary plus config, and chart-releaser assumes GitHub Pages +
# GitHub Releases, while this repo has two charts and builds on Volcengine CP. Ten lines beat a
# toolchain here.
#
# NOTE values.yaml counts as a chart change: values ship inside the package, so changing a
# default changes what `helm pull` hands the next person.
#
#   scripts/check-chart-version.sh [<base-ref>]      # default: HEAD~1
#
# In CI compare against the previous commit; locally, pass a branch point (e.g. origin/main).
set -euo pipefail

BASE="${1:-HEAD~1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

git rev-parse --verify --quiet "$BASE" >/dev/null || {
  echo "check-chart-version: base ref '$BASE' not found — skipping (shallow clone?)" >&2
  exit 0
}

fail=0
for chart in charts/*/; do
  [ -f "${chart}Chart.yaml" ] || continue
  name="$(basename "$chart")"

  # Anything under the chart dir counts, including values.yaml and values-*.yaml.
  if git diff --quiet "$BASE" -- "$chart"; then
    continue
  fi

  # A bumped version shows up as an added `version:` line in Chart.yaml. Match the top-level key
  # only — `appVersion:` is informational here (the image tag comes from values at deploy time),
  # so bumping it must NOT satisfy this check.
  if git diff "$BASE" -- "${chart}Chart.yaml" | grep -qE '^\+version:'; then
    new="$(grep -E '^version:' "${chart}Chart.yaml" | head -1 | awk '{print $2}')"
    echo "  ok    $name -> $new"
  else
    old="$(grep -E '^version:' "${chart}Chart.yaml" | head -1 | awk '{print $2}')"
    echo "  FAIL  $name changed but version is still $old"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  cat >&2 <<'MSG'

A chart changed without a version bump. Edit its Chart.yaml `version:` (semver) before pushing,
or the registry ends up with two different charts under one tag.
MSG
  exit 1
fi
echo "check-chart-version: ok"
