#!/usr/bin/env bash
# Evaluate the baselines on the CSG-Fusion ScanNet++ test set (masked metrics, overlaps 0.9 / 0.7 / 0.5 / 0.3).
# usage: bash scripts/eval/eval_baselines.sh [method ...] [-- extra args for python -m scripts.eval.baselines]
#   No method given: every method below, with the checkpoints of scripts/eval/baselines/__init__.py (see README).
#   e.g. bash scripts/eval/eval_baselines.sh noposplat -- --save-images
#        bash scripts/eval/eval_baselines.sh csg_fusion     # CSG-Fusion itself, same protocol
# env: CSG_FUSION_CKPT   CSG-Fusion checkpoint, default checkpoint/csg_fusion.ckpt (the released model)
set -euo pipefail
cd "$(dirname "$0")/../.."
methods=(); extra=()
while [[ $# -gt 0 ]]; do
    if [[ $1 == -- ]]; then shift; extra=("$@"); break; fi
    methods+=("$1"); shift
done
[[ ${#methods[@]} -eq 0 ]] && methods=(pixelsplat mvsplat noposplat splatt3r depthsplat anysplat yonosplat spfsplat spfsplatv2_l csg_fusion)
for m in "${methods[@]}"; do
    ckpt=(); [[ $m == csg_fusion && -n ${CSG_FUSION_CKPT:-} ]] && ckpt=(--checkpoint "$CSG_FUSION_CKPT")
    python -m scripts.eval.baselines --method "$m" ${ckpt[@]+"${ckpt[@]}"} ${extra[@]+"${extra[@]}"}
done
