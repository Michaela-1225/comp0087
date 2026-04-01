#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/small.yaml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}"
  exit 1
fi

LOG_FILE=$(python - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = sys.argv[1]

with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

logging_cfg = config.get("logging", {})
training_cfg = config.get("training", {})

log_file = logging_cfg.get("train_log_file")
if not log_file:
    output_dir = training_cfg.get("output_dir", "outputs/default_run")
    log_file = str(Path(output_dir) / "logs" / "train.log")

print(log_file)
PY
)

mkdir -p "$(dirname "${LOG_FILE}")"

echo "========================================"
echo "Running Stage 1 training"
echo "Config file: ${CONFIG_PATH}"
echo "Train log : ${LOG_FILE}"
echo "========================================"

python -u train_sft.py --config "${CONFIG_PATH}" 2>&1 | tee "${LOG_FILE}"
