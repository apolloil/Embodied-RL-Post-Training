#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
用法:
  bash scripts/launch/state_based.sh <Serial_Bro|Serial_TF|Parallel_Bro|Parallel_TF|all> [Hydra overrides...]

环境变量:
  PYTHON_BIN          Python 可执行文件，默认 python
  RUN_MODE            foreground 或 background；单任务默认 foreground，all 默认 background
  LOG_DIR             后台模式日志目录，默认 logs/state
  GPU                 单任务默认 GPU；也可单独指定 GPU_SERIAL_BRO / GPU_SERIAL_TF / GPU_PARALLEL_BRO / GPU_PARALLEL_TF
  IS_PREFIX           实验名覆盖，默认与算法名一致

示例:
  bash scripts/launch/state_based.sh Serial_TF
  GPU=0 bash scripts/launch/state_based.sh Parallel_Bro
  GPU_SERIAL_BRO=0 GPU_SERIAL_TF=1 GPU_PARALLEL_BRO=2 GPU_PARALLEL_TF=3 \
    bash scripts/launch/state_based.sh all
EOF
}

if [[ $# -lt 1 ]] || [[ "${1}" == "-h" ]] || [[ "${1}" == "--help" ]]; then
  usage
  exit 0
fi

algo="$1"
shift

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/state}"
RUN_MODE="${RUN_MODE:-}"
IS_PREFIX="${IS_PREFIX:-}"

if [[ -z "${RUN_MODE}" ]]; then
  if [[ "${algo}" == "all" ]]; then
    RUN_MODE="background"
  else
    RUN_MODE="foreground"
  fi
fi

declare -A ENTRYPOINTS=(
  [Serial_Bro]="experiments/state/Serial_Bro.py"
  [Serial_TF]="experiments/state/Serial_TF.py"
  [Parallel_Bro]="experiments/state/Parallel_Bro.py"
  [Parallel_TF]="experiments/state/Parallel_TF.py"
)

gpu_for() {
  case "$1" in
    Serial_Bro) printf '%s' "${GPU_SERIAL_BRO:-${GPU:-}}" ;;
    Serial_TF) printf '%s' "${GPU_SERIAL_TF:-${GPU:-}}" ;;
    Parallel_Bro) printf '%s' "${GPU_PARALLEL_BRO:-${GPU:-}}" ;;
    Parallel_TF) printf '%s' "${GPU_PARALLEL_TF:-${GPU:-}}" ;;
    *) return 1 ;;
  esac
}

launch_one() {
  local target="$1"
  shift

  local entry="${ENTRYPOINTS[$target]:-}"
  if [[ -z "${entry}" ]]; then
    echo "不支持的算法: ${target}" >&2
    exit 1
  fi

  local run_name
  if [[ -n "${IS_PREFIX}" ]]; then
    if [[ "${algo}" == "all" ]]; then
      run_name="${IS_PREFIX}_${target}"
    else
      run_name="${IS_PREFIX}"
    fi
  else
    run_name="${target}"
  fi
  local gpu
  gpu="$(gpu_for "${target}")"

  local -a cmd=(
    "${PYTHON_BIN}"
    "${REPO_ROOT}/${entry}"
    "IS=${run_name}"
    "env.suite_name=State_Based"
  )
  cmd+=("$@")

  if [[ "${RUN_MODE}" == "background" ]]; then
    mkdir -p "${LOG_DIR}"
    local log_file="${LOG_DIR}/${run_name}.log"
    if [[ -n "${gpu}" ]]; then
      nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >"${log_file}" 2>&1 &
    else
      nohup "${cmd[@]}" >"${log_file}" 2>&1 &
    fi
    echo "[background] ${target} -> PID=$! log=${log_file}"
  else
    echo "[foreground] ${target}"
    if [[ -n "${gpu}" ]]; then
      env CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
    else
      "${cmd[@]}"
    fi
  fi
}

if [[ "${algo}" == "all" ]]; then
  launch_one Serial_Bro "$@"
  launch_one Serial_TF "$@"
  launch_one Parallel_Bro "$@"
  launch_one Parallel_TF "$@"
else
  launch_one "${algo}" "$@"
fi
