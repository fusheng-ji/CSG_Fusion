#!/bin/bash
# Stitch the ScanNet++ comparison images of all baselines and CSG-Fusion for every overlap level.
#
# Reads the renders of the uniform baseline harness (run each method with --save-images first):
#   python -m scripts.eval.baselines --method <m> --checkpoint <ckpt> --save-images
#   -> $RESULTS_ROOT/<m>/scannetpp/overlap_<o>/images/
# Context input views are read from $DATA_ROOT (ScanNet++). Output: $OUTPUT_ROOT/overlap_<o>/
# All paths are relative to the repository root; override them with RESULTS_ROOT=... DATA_ROOT=... OUTPUT_ROOT=...
cd "$(dirname "$0")/../.."

RESULTS_ROOT="${RESULTS_ROOT:-results/baselines}"
DATA_ROOT="${DATA_ROOT:-data/scannetpp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/comparison/scannetpp}"

# Display order (the last one is CSG-Fusion)
METHODS=(pixelsplat mvsplat noposplat splatt3r csg_fusion)
OVERLAPS=("0.3" "0.5" "0.7" "0.9")

echo "Stitching comparison images for all overlap levels..."

for overlap in "${OVERLAPS[@]}"; do
    echo "======================================="
    echo "Overlap ${overlap}"
    echo "======================================="

    python scripts/viz/image_stitching_algorithm.py \
        --overlap "$overlap" \
        --methods "${METHODS[@]}" \
        --results_root "$RESULTS_ROOT" \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_ROOT" \
        --target_size 512

    echo "Overlap ${overlap} done: $OUTPUT_ROOT/overlap_${overlap}"
    echo ""
done

echo "======================================="
echo "All overlap levels done. Results:"
for overlap in "${OVERLAPS[@]}"; do
    echo "  - $OUTPUT_ROOT/overlap_${overlap}"
done
echo "======================================="
