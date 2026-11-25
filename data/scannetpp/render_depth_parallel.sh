#!/usr/bin/env bash
# Run prepare_processed.py with several renderpy workers on disjoint scene shards.
# Test (nvs_sem_val) scenes go first. Resumable: finished frames are skipped.
#
# usage: bash data/scannetpp/render_depth_parallel.sh [--workers N] [--gpus 0,1] [--follow]
#   --workers N   parallel renderpy processes (default 6)
#   --gpus LIST   GPUs to spread the workers over, round-robin (default 0)
#   --follow      keep polling every 5 min for newly downloaded scenes until every coverage scene is done
# env: PYTHON (python with renderpy, default `python`)
# A scene counts as downloaded once its mesh exists, which is the last asset in the download order.
set -uo pipefail
cd "$(dirname "$0")/../.."                       # repo root
WORKERS=6; GPUS=0; FOLLOW=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers) WORKERS=$2; shift 2 ;;
        --gpus) GPUS=$2; shift 2 ;;
        --follow) FOLLOW=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done
PY=${PYTHON:-python}
IFS=, read -ra GPU_LIST <<< "$GPUS"
# headless NVIDIA: make EGL pick the NVIDIA vendor library
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-$NV_EGL}

D=data/scannetpp
# coverage scenes that are in the v1 train / val splits (all 276 of them once 280b83fcf3.json is deleted, see README)
ALL=$(ls $D/coverage | sed 's/.json$//' | grep -xFf <(cat $D/raw/splits/nvs_sem_{train,val}.txt))
VAL=$(cat $D/raw/splits/nvs_sem_val.txt)
TOTAL=$(echo "$ALL" | wc -l)
complete() {   # as many depth maps as images
    local ni nd
    ni=$(ls $D/raw/data/$1/dslr/resized_undistorted_images 2>/dev/null | wc -l)
    nd=$(ls $D/processed/$1/dslr/undistorted_depths 2>/dev/null | wc -l)
    [ "$ni" -gt 0 ] && [ "$ni" = "$nd" ]
}
while true; do
    todo=()
    for s in $(echo "$ALL" | grep -xFf <(echo "$VAL")) $(echo "$ALL" | grep -vxFf <(echo "$VAL")); do
        [ -f $D/raw/data/$s/scans/mesh_aligned_0.05.ply ] && ! complete $s && todo+=("$s")
    done
    if [ ${#todo[@]} -gt 0 ]; then
        echo "=== $(date '+%F %T') rendering ${#todo[@]} scenes with $WORKERS workers on GPU(s) $GPUS ==="
        for ((w = 0; w < WORKERS; w++)); do
            shard=(); for ((i = w; i < ${#todo[@]}; i += WORKERS)); do shard+=("${todo[$i]}"); done
            [ ${#shard[@]} -gt 0 ] || continue
            CUDA_VISIBLE_DEVICES=${GPU_LIST[$((w % ${#GPU_LIST[@]}))]} $PY $D/prepare_processed.py --scenes "${shard[@]}" 2>&1 \
                | tr '\r' '\n' | grep --line-buffered -v "it\]$\|s/it\]$\|^\s*$\|Destroy EGL\|Detected\|Using \|Setup the" \
                | sed -u "s/^/[w$w] /" &
        done
        wait
    fi
    done_n=0; for s in $ALL; do complete $s && done_n=$((done_n + 1)); done
    echo "=== $(date '+%F %T') scenes with complete depth: $done_n / $TOTAL ==="
    [ $FOLLOW = 1 ] && [ "$done_n" -lt "$TOTAL" ] || break
    sleep 300
done
