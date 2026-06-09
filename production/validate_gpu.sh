#!/usr/bin/env bash
set -euo pipefail

run_id="${1:-all}"
shift || true
extra_args=()
has_steps=0
has_resolution_tier=0
validate_only_requested=0
full_run=0
for arg in "$@"; do
  case "$arg" in
    --smoke)
      ;;
    --full)
      full_run=1
      ;;
    --steps|--steps=*)
      has_steps=1
      extra_args+=("$arg")
      ;;
    --resolution-tier|--resolution-tier=*)
      has_resolution_tier=1
      extra_args+=("$arg")
      ;;
    --validate-only)
      validate_only_requested=1
      extra_args+=("$arg")
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
logs_root="${JAXFUN_VALIDATE_LOGS_DIR:-logs}"
heavy_resolution_tier="${JAXFUN_VALIDATE_RESOLUTION_TIER:-start}"
heavy_steps="${JAXFUN_VALIDATE_HEAVY_STEPS:-2}"

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
  all, heavy   execute production/runs/*.json as bounded smoke by default
               (default resolution tier start, default 2 steps)
  cheap        run the seven non-pipe cheap golden comparisons
  dns          run the four committed linear-window DNS golden comparisons
  dns-pcf      run the PCF primitive linear-window DNS golden comparisons
  dns-tc       run the Taylor-Couette linear-window DNS golden comparisons
  problem_id   execute production/runs/<problem_id>.json as bounded smoke, or
               compare production/examples/<problem_id>.json if no run spec exists

Heavy-run options:
  --full         run the checked-in spec without smoke defaults
  --validate-only keep metadata-only validation for production/runs specs
  --smoke        accepted for compatibility; smoke is the default
USAGE
}

run_with_log() {
  local id="$1"
  shift
  local log="${logs_root}/${id}.log"
  mkdir -p "$logs_root"
  local started_epoch
  started_epoch="$(date +%s)"
  {
    echo "timestamp=${timestamp}"
    echo "mode=${run_id}"
    echo "problem_id=${id}"
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
  } > "$log"

  set +e
  "$@" >> "$log" 2>&1
  local status="$?"
  set -e
  local finished_epoch
  finished_epoch="$(date +%s)"
  {
    echo "exit_status=${status}"
    echo "duration_seconds=$((finished_epoch - started_epoch))"
    echo "completed_at_utc=$(date -u +%Y%m%dT%H%M%SZ)"
  } >> "$log"
  if [[ "$status" -ne 0 ]]; then
    echo "validation failed for ${id} with exit ${status}; see ${log}" >&2
    tail -n 80 "$log" >&2 || true
    return "$status"
  fi
}

run_heavy_spec() {
  local id="$1"
  local config="production/runs/${id}.json"
  if [[ ! -f "$config" ]]; then
    echo "missing run spec: $config" >&2
    exit 1
  fi
  local out="runs/${id}/${timestamp}"
  mkdir -p "$out"
  local heavy_args=()
  if [[ "$validate_only_requested" -eq 0 && "$full_run" -eq 0 ]]; then
    if [[ "$has_resolution_tier" -eq 0 && -n "$heavy_resolution_tier" ]]; then
      heavy_args+=(--resolution-tier "$heavy_resolution_tier")
    fi
    if [[ "$has_steps" -eq 0 && -n "$heavy_steps" ]]; then
      heavy_args+=(--steps "$heavy_steps")
    fi
  fi
  run_with_log "$id" \
    timeout "${timeout_seconds}s" "$python_bin" production/run_problem.py \
      --config "$config" \
      --out "$out" \
      --device auto \
      "${heavy_args[@]}" \
      "${extra_args[@]}"
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
  run_with_log "$id" \
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
      run_heavy_spec "$id"
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
      run_heavy_spec "$run_id"
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
