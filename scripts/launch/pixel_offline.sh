#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
用法:
  bash scripts/launch/pixel_offline.sh <OFF_BRC_BC|OFF_TD3BC|OFF_CalQL|OFF_EDAC|all> [Hydra overrides...]

环境变量:
  PYTHON_BIN           Python 可执行文件，默认 python
  RUN_MODE             foreground 或 background；单任务默认 foreground，all 默认 background
  LOG_DIR              后台模式日志目录，默认 logs/pixel_offline
  GPU                  单任务默认 GPU；也可为每个算法单独指定 GPU_OFF_BRC_BC / GPU_OFF_TD3BC / GPU_OFF_CalQL / GPU_OFF_EDAC
  IS_PREFIX            实验名覆盖，默认与算法名一致

示例:
  bash scripts/launch/pixel_offline.sh OFF_BRC_BC
  GPU=0 bash scripts/launch/pixel_offline.sh OFF_CalQL train.rl.batch_size=512
  GPU_OFF_BRC_BC=0 GPU_OFF_TD3BC=1 GPU_OFF_CalQL=2 GPU_OFF_EDAC=3 \
    bash scripts/launch/pixel_offline.sh all
EOF
}

if [[ $# -lt 1 ]] || [[ "${1}" == "-h" ]] || [[ "${1}" == "--help" ]]; then
  usage
  exit 0
fi

algo="$1"
shift

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/pixel_offline}"
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
  [OFF_BRC_BC]="experiments/pixel/offline/OFF_BRC_BC.py"
  [OFF_TD3BC]="experiments/pixel/offline/OFF_TD3BC.py"
  [OFF_CalQL]="experiments/pixel/offline/OFF_CalQL.py"
  [OFF_EDAC]="experiments/pixel/offline/OFF_EDAC.py"
)

default_extra_args() {
  case "$1" in
    OFF_EDAC) printf '%s\n' "train.rl.critic.hidden_dim=512" ;;
    *) ;;
  esac
}

gpu_for() {
  case "$1" in
    OFF_BRC_BC) printf '%s' "${GPU_OFF_BRC_BC:-${GPU:-}}" ;;
    OFF_TD3BC) printf '%s' "${GPU_OFF_TD3BC:-${GPU:-}}" ;;
    OFF_CalQL) printf '%s' "${GPU_OFF_CalQL:-${GPU:-}}" ;;
    OFF_EDAC) printf '%s' "${GPU_OFF_EDAC:-${GPU:-}}" ;;
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
    "env.suite_name=Pixel_Offline"
  )

  while IFS= read -r extra; do
    [[ -n "${extra}" ]] && cmd+=("${extra}")
  done < <(default_extra_args "${target}")

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
  launch_one OFF_BRC_BC "$@"
  launch_one OFF_TD3BC "$@"
  launch_one OFF_CalQL "$@"
  launch_one OFF_EDAC "$@"
else
  launch_one "${algo}" "$@"
fi
