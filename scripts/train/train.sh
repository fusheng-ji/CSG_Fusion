#!/usr/bin/env bash
# Train + test CSG-Fusion on this machine.
#
# Usage: bash scripts/train/train.sh [configs/<exp>.yaml] [key=value ...]
#   bash scripts/train/train.sh                                            # CSG-Fusion (default config)
#   GPUS=0,1 bash scripts/train/train.sh configs/ablations/dist_threshold_0.05.yaml opt.epochs=10
#
# Environment:
#   GPUS    GPUs to use (CUDA_VISIBLE_DEVICES), default 0. Several GPUs -> DDP.
#   PYTHON  Python interpreter, default `python`.
# Output goes to results/<name>/ (see the config) and a copy of the console log to logs/.
#
# Experiment configs:
#   configs/csg_fusion.yaml        (default: the published model's recipe)
#   configs/base.yaml              (base of all configs; no context / MASt3R loss)
#   configs/csg_fusion_vggt.yaml   (VGGT backbone)
#   configs/ablations/*.yaml       (ablations and threshold sweep)
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG=configs/csg_fusion.yaml
if [[ $# -gt 0 && $1 == *.yaml ]]; then CONFIG=$1; shift; fi

export CUDA_VISIBLE_DEVICES=${GPUS:-0}
mkdir -p logs
LOG=logs/$(basename "$CONFIG" .yaml)_$(date +%Y%m%d-%H%M%S).log
echo "config: $CONFIG | GPUs: $CUDA_VISIBLE_DEVICES | log: $LOG"
${PYTHON:-python} main.py "$CONFIG" "$@" 2>&1 | tee "$LOG"
