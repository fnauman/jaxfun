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
timeout_seconds="${JAXFUN_VALIDATE_TIMEOUT_SECONDS:-1800}"

cheap_parity_ids=(
  pcf_hydro_laminar_v1
  channel_poiseuille_hydro_v1
  pcf_mhd_conducting_v1
  pcf_mri_shearbox_v1
  taylor_couette_hydro_v1
  taylor_couette_mhd_conducting_v1
  taylor_couette_mhd_insulating_v1
)

pcf_dns_parity_ids=(
  pcf_hydro_primitive_dns_v1
  pcf_mri_primitive_dns_v1
)

tc_dns_parity_ids=(
  taylor_couette_hydro_dns_v1
  taylor_couette_mhd_dns_v1
)

dns_parity_ids=("${pcf_dns_parity_ids[@]}" "${tc_dns_parity_ids[@]}")

usage() {
  cat >&2 <<'USAGE'
usage: production/validate_gpu.sh [all|heavy|cheap|dns|dns-pcf|dns-tc|problem_id] [run_problem args...]

Modes:
  all, heavy   validate production/runs/*.json without executing unwired heavy solvers
  cheap        run the seven non-pipe cheap golden comparisons
  dns          run the four committed linear-window DNS golden comparisons
  dns-pcf      run the PCF primitive linear-window DNS golden comparisons
  dns-tc       run the Taylor-Couette linear-window DNS golden comparisons
  problem_id   validate production/runs/<problem_id>.json, or compare
               production/examples/<problem_id>.json if no run spec exists
USAGE
}

run_validate_only() {
  local id="$1"
  local config="production/runs/${id}.json"
  if [[ ! -f "$config" ]]; then
    echo "missing run spec: $config" >&2
    exit 1
  fi
  local out="runs/${id}/${timestamp}"
  mkdir -p "$out"
  timeout "${timeout_seconds}s" "$python_bin" production/run_problem.py \
    --config "$config" \
    --out "$out" \
    --device auto \
    --validate-only "${extra_args[@]}"
}

run_compare_golden() {
  local id="$1"
  local config="production/examples/${id}.json"
  if [[ ! -f "$config" ]]; then
    echo "missing example spec: $config" >&2
    exit 1
  fi
  local out="runs/${id}/${timestamp}"
  mkdir -p "$out"
  timeout "${timeout_seconds}s" "$python_bin" production/run_problem.py \
    --config "$config" \
    --out "$out" \
    --device auto \
    --compare-golden "${extra_args[@]}"
}

case "$run_id" in
  all|heavy)
    mapfile -t run_ids < <(find production/runs -maxdepth 1 -name '*.json' -printf '%f\n' | sed 's/\.json$//' | sort)
    for id in "${run_ids[@]}"; do
      run_validate_only "$id"
    done
    ;;
  cheap|parity-cheap)
    for id in "${cheap_parity_ids[@]}"; do
      run_compare_golden "$id"
    done
    echo "pipe_hagen_poiseuille_v1 and pipe_womersley_v1 are skipped: pipe hydro is parity_pending until the axis-regular radial basis lands"
    ;;
  dns|parity-dns)
    for id in "${dns_parity_ids[@]}"; do
      run_compare_golden "$id"
    done
    ;;
  dns-pcf|pcf-dns|parity-dns-pcf)
    for id in "${pcf_dns_parity_ids[@]}"; do
      run_compare_golden "$id"
    done
    ;;
  dns-tc|tc-dns|parity-dns-tc)
    for id in "${tc_dns_parity_ids[@]}"; do
      run_compare_golden "$id"
    done
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    if [[ -f "production/runs/${run_id}.json" ]]; then
      run_validate_only "$run_id"
    elif [[ -f "production/examples/${run_id}.json" ]]; then
      run_compare_golden "$run_id"
    else
      echo "unknown validation mode or problem_id: $run_id" >&2
      usage
      exit 1
    fi
    ;;
esac

"$python_bin" -m production.report --runs-root runs --out runs/_report
echo "wrote runs/_report/results.json"
