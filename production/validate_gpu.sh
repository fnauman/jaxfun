#!/usr/bin/env bash
set -euo pipefail

run_id="${1:-all}"
shift || true
extra_args=()
for arg in "$@"; do
  case "$arg" in
    --smoke)
      ;;
    *)
      extra_args+=("$arg")
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
python_bin="${PYTHON:-.venv/bin/python}"

if [[ "$run_id" == "all" ]]; then
  mapfile -t run_ids < <(find production/runs -maxdepth 1 -name '*.json' -printf '%f\n' | sed 's/\.json$//' | sort)
else
  run_ids=("$run_id")
fi

for id in "${run_ids[@]}"; do
  config="production/runs/${id}.json"
  if [[ ! -f "$config" ]]; then
    echo "missing run spec: $config" >&2
    exit 1
  fi
  out="runs/${id}/${timestamp}"
  mkdir -p "$out"
  "$python_bin" production/run_problem.py \
    --config "$config" \
    --out "$out" \
    --device auto \
    --validate-only "${extra_args[@]}"
done

"$python_bin" -m production.report --runs-root runs --out runs/_report
echo "wrote runs/_report/results.json"
