# Adapted from pixelSplat (https://github.com/dcharatan/pixelsplat), MIT License,
# Copyright (c) 2023 David Charatan, Sizhe Li, Andrea Tagliasacchi and Vincent Sitzmann. Modified for CSG-Fusion.
"""Camera-geometry helpers of the renderer (pixelSplat's ``projection.py``): field of view from normalised
intrinsics and the 3x3 covariance upper-triangle layout of the rasterizer."""
import torch
from einops import einsum


def get_fov(intrinsics):
    """(B, 3, 3) normalised intrinsics -> (B, 2) horizontal and vertical field of view in radians."""
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)


_COV_UPPER = {}


def cov_upper_triangle(cov):
    """(..., 3, 3) -> (..., 6) entries (0,0) (0,1) (0,2) (1,1) (1,2) (2,2), the cov3D_precomp layout.

    Same values as cov[..., row, col] with row, col = triu_indices(3, 3), but one index_select for all
    views (cheap backward) instead of per-view advanced indexing with CPU index tensors (an H2D copy and
    stream sync per view, and a sort-based index_put in backward)."""
    idx = _COV_UPPER.get(cov.device)
    if idx is None:
        idx = _COV_UPPER[cov.device] = torch.tensor([0, 1, 2, 4, 5, 8], device=cov.device)
    return cov.flatten(-2).index_select(-1, idx)
