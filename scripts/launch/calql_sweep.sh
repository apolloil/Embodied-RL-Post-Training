#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
用法:
  CALQL_PAIRS="0.1:4.693 0.2:5.0" bash scripts/launch/calql_sweep.sh [Hydra overrides...]

环境变量:
  CALQL_PAIRS   多组 "weight:gap"，默认 "0.1:4.693"
  PYTHON_BIN    Python 可执行文件，默认 python
  RUN_MODE      foreground 或 background，默认 background
  LOG_DIR       后台模式日志目录，默认 logs/calql_sweep
  GPU           运行使用的 GPU
  IS_PREFIX     实验名前缀，默认 OFF_CalQL

示例:
  CALQL_PAIRS="0.05:4.0 0.1:4.693 0.2:5.0" bash scripts/launch/calql_sweep.sh
EOF
}

if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-background}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/calql_sweep}"
CALQL_PAIRS="${CALQL_PAIRS:-0.1:4.693}"
IS_PREFIX="${IS_PREFIX:-OFF_CalQL}"

mkdir -p "${LOG_DIR}"

for pair in ${CALQL_PAIRS}; do
  weight="${pair%%:*}"
  gap="${pair#*:}"
  if [[ -z "${weight}" ]] || [[ -z "${gap}" ]] || [[ "${weight}" == "${gap}" ]]; then
    echo "非法 CALQL_PAIRS 条目: ${pair}，应为 weight:gap" >&2
    exit 1
  fi

  tag="${IS_PREFIX}_w${weight}_g${gap}"
  cmd=(
    "${PYTHON_BIN}"
    "${REPO_ROOT}/experiments/pixel/offline/OFF_CalQL.py"
    "IS=${tag}"
    "env.suite_name=Pixel_Offline"
  )
  cmd+=("$@")

  if [[ "${RUN_MODE}" == "background" ]]; then
    log_file="${LOG_DIR}/${tag}.log"
    if [[ -n "${GPU:-}" ]]; then
      nohup env CQL_WEIGHT="${weight}" CQL_GAP="${gap}" CUDA_VISIBLE_DEVICES="${GPU}" \
        "${cmd[@]}" >"${log_file}" 2>&1 &
    else
      nohup env CQL_WEIGHT="${weight}" CQL_GAP="${gap}" \
        "${cmd[@]}" >"${log_file}" 2>&1 &
    fi
    echo "[background] ${tag} -> PID=$! log=${log_file}"
  else
    echo "[foreground] ${tag}"
    if [[ -n "${GPU:-}" ]]; then
      env CQL_WEIGHT="${weight}" CQL_GAP="${gap}" CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}"
    else
      env CQL_WEIGHT="${weight}" CQL_GAP="${gap}" "${cmd[@]}"
    fi
  fi
done
