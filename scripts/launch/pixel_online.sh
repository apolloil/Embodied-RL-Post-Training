#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
用法:
  bash scripts/launch/pixel_online.sh <ON_BRC|ON_BRC_BC|ON_CalQL|ON_SO2|ON_TD3BC> [Hydra overrides...]

说明:
  ON_BRC 为从零开始训练，不需要 checkpoint。
  其余 4 个脚本默认需要通过 SNAPSHOT_PATH 或 train.rl.rl_snapshot=... 指定初始化 checkpoint。

环境变量:
  PYTHON_BIN       Python 可执行文件，默认 python
  RUN_MODE         foreground 或 background，默认 foreground
  LOG_DIR          后台模式日志目录，默认 logs/pixel_online
  GPU              默认 GPU；也可单独指定 GPU_ON_BRC / GPU_ON_BRC_BC / GPU_ON_CalQL / GPU_ON_SO2 / GPU_ON_TD3BC
  SNAPSHOT_PATH    微调初始 checkpoint 路径
  IS_PREFIX        实验名覆盖，默认与算法名一致

示例:
  bash scripts/launch/pixel_online.sh ON_BRC
  SNAPSHOT_PATH=/path/to/snapshot.pt bash scripts/launch/pixel_online.sh ON_BRC_BC
  GPU=0 bash scripts/launch/pixel_online.sh ON_TD3BC train.rl.batch_size=512
EOF
}

if [[ $# -lt 1 ]] || [[ "${1}" == "-h" ]] || [[ "${1}" == "--help" ]]; then
  usage
  exit 0
fi

algo="$1"
shift

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-foreground}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/pixel_online}"
IS_PREFIX="${IS_PREFIX:-}"

declare -A ENTRYPOINTS=(
  [ON_BRC]="experiments/pixel/online/ON_BRC.py"
  [ON_BRC_BC]="experiments/pixel/online/ON_BRC_BC.py"
  [ON_CalQL]="experiments/pixel/online/ON_CalQL.py"
  [ON_SO2]="experiments/pixel/online/ON_SO2.py"
  [ON_TD3BC]="experiments/pixel/online/ON_TD3BC.py"
)

gpu_for() {
  case "$1" in
    ON_BRC) printf '%s' "${GPU_ON_BRC:-${GPU:-}}" ;;
    ON_BRC_BC) printf '%s' "${GPU_ON_BRC_BC:-${GPU:-}}" ;;
    ON_CalQL) printf '%s' "${GPU_ON_CalQL:-${GPU:-}}" ;;
    ON_SO2) printf '%s' "${GPU_ON_SO2:-${GPU:-}}" ;;
    ON_TD3BC) printf '%s' "${GPU_ON_TD3BC:-${GPU:-}}" ;;
    *) return 1 ;;
  esac
}

needs_snapshot() {
  [[ "$1" != "ON_BRC" ]]
}

has_snapshot_override() {
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == train.rl.rl_snapshot=* ]]; then
      return 0
    fi
  done
  return 1
}

entry="${ENTRYPOINTS[$algo]:-}"
if [[ -z "${entry}" ]]; then
  echo "不支持的算法: ${algo}" >&2
  exit 1
fi

if needs_snapshot "${algo}" && ! has_snapshot_override "$@" && [[ -z "${SNAPSHOT_PATH:-}" ]]; then
  echo "算法 ${algo} 需要初始化 checkpoint，请设置 SNAPSHOT_PATH 或传入 train.rl.rl_snapshot=/path/to/snapshot.pt" >&2
  exit 1
fi

run_name="${IS_PREFIX:-${algo}}"
gpu="$(gpu_for "${algo}")"

cmd=(
  "${PYTHON_BIN}"
  "${REPO_ROOT}/${entry}"
  "IS=${run_name}"
  "env.suite_name=Pixel_Online"
)

if [[ -n "${SNAPSHOT_PATH:-}" ]] && ! has_snapshot_override "$@"; then
  cmd+=("train.rl.rl_snapshot=${SNAPSHOT_PATH}")
fi

cmd+=("$@")

if [[ "${RUN_MODE}" == "background" ]]; then
  mkdir -p "${LOG_DIR}"
  log_file="${LOG_DIR}/${run_name}.log"
  if [[ -n "${gpu}" ]]; then
    nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >"${log_file}" 2>&1 &
  else
    nohup "${cmd[@]}" >"${log_file}" 2>&1 &
  fi
  echo "[background] ${algo} -> PID=$! log=${log_file}"
else
  echo "[foreground] ${algo}"
  if [[ -n "${gpu}" ]]; then
    env CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
fi
