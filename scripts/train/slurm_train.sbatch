#!/bin/bash
# Train one config on a Slurm cluster (one node; the Lightning Trainer uses every GPU the job is given).
# usage: sbatch --gres=gpu:<n> --export=ALL,CONFIG=configs/<x>.yaml,ARGS="<key=value ...>" scripts/train/slurm_train.sbatch
# env:   PYTHON (default: python), TORCH_HOME (torch hub cache; LPIPS needs the AlexNet weights in hub/checkpoints)
# The resource lines below are examples: adjust them to your cluster.
#SBATCH --job-name=csg_fusion
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
PY=${PYTHON:-python}
export WANDB_MODE=${WANDB_MODE:-disabled}
echo "[$(date '+%F %T')] GPUs ${CUDA_VISIBLE_DEVICES:-?} config=$CONFIG args=${ARGS:-}"
$PY main.py "$CONFIG" ${ARGS:-}
status=$?
echo "[$(date '+%F %T')] exit $status"
exit $status
