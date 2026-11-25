#!/usr/bin/env bash
# Build and install the CUDA extensions CSG-Fusion needs, at the exact versions used for the paper.
# Run inside the activated `csg_fusion` (or `csg_fusion_cu128`) env:
#   bash scripts/install_rasterizers.sh [--no-rope]
#   TORCH_CUDA_ARCH_LIST="8.6;9.0" bash scripts/install_rasterizers.sh
#
# TORCH_CUDA_ARCH_LIST (passed through to the PyTorch extension build of diff_gaussian_rasterization): set it
# to your GPU's compute capability, e.g. 8.6, 9.0, 10.0 (several separated by ';'); unset = the GPUs of this
# machine. curope ignores it and always builds for every architecture the installed PyTorch was built for.
#
#  1. diff_gaussian_rasterization       git submodule third_party/diff-gaussian-rasterization
#       (fusheng-ji/diff-gaussian-rasterization: dcharatan's modified rasterizer + skipping zero-opacity,
#        i.e. fused / masked, Gaussians; used for all results)
#  2. curope (default on)               CUDA RoPE kernels from CroCo v2 (opt-in with perf.rope=cuda)
set -euo pipefail
cd "$(dirname "$0")/.."

ROPE=1
for arg in "$@"; do
    case $arg in
        --no-rope) ROPE=0 ;;
        *) echo "unknown option: $arg" >&2; exit 1 ;;
    esac
done

if [[ -n ${TORCH_CUDA_ARCH_LIST:-} ]]; then
    export TORCH_CUDA_ARCH_LIST
    echo "==> TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
else
    echo "==> TORCH_CUDA_ARCH_LIST not set: building for the GPUs of this machine"
fi

echo "==> diff_gaussian_rasterization (submodule)"
git submodule update --init --recursive third_party/diff-gaussian-rasterization
pip install --no-build-isolation third_party/diff-gaussian-rasterization

if [[ $ROPE == 1 ]]; then
    echo "==> curope (RoPE CUDA kernels)"
    (cd src/mast3r_src/dust3r/croco/models/curope && python setup.py build_ext --inplace)
fi

python -c "import diff_gaussian_rasterization; print('diff_gaussian_rasterization OK')"
