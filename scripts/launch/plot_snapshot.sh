#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
用法:
  bash scripts/launch/plot_snapshot.sh <snapshot_path> [output_name] [Hydra overrides...]

环境变量:
  PYTHON_BIN        Python 可执行文件，默认 python
  IS_PIXEL_BASED    true 或 false，默认 true
  OUTPUT_NAME       输出目录名；若未提供第二个位置参数，则默认使用 checkpoint 文件名
  SEED              可视化采样随机种子，默认沿用脚本内部默认值

示例:
  bash scripts/launch/plot_snapshot.sh /path/to/snapshot_latest.pt
  IS_PIXEL_BASED=false bash scripts/launch/plot_snapshot.sh /path/to/state.pt state_debug
  bash scripts/launch/plot_snapshot.sh /path/to/off_edac.pt EDAC \
    train.rl.critic.hidden_dim=512 train.rl.critic.ensemble_size=10
EOF
}

if [[ $# -lt 1 ]] || [[ "${1}" == "-h" ]] || [[ "${1}" == "--help" ]]; then
  usage
  exit 0
fi

snapshot_path="$1"
shift

default_output_name="$(basename "${snapshot_path}")"
default_output_name="${default_output_name%.pt}"
output_name="${1:-${OUTPUT_NAME:-${default_output_name}}}"
if [[ $# -ge 1 ]]; then
  shift
fi

if [[ ! -f "${snapshot_path}" ]]; then
  echo "checkpoint 不存在: ${snapshot_path}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
IS_PIXEL_BASED="${IS_PIXEL_BASED:-true}"

env \
  IS_PIXEL_BASED="${IS_PIXEL_BASED}" \
  SNAPSHOT_PATH="${snapshot_path}" \
  OUTPUT_NAME="${output_name}" \
  SEED="${SEED:-42}" \
  "${PYTHON_BIN}" \
  "${REPO_ROOT}/tools/Plot_Snapshot.py" \
  hydra.output_subdir=null \
  hydra.run.dir=. \
  "$@"
