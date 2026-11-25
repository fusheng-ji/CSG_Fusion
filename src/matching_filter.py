"""Matching-based Gaussian Filter (paper Sec. 4.1) on SciPy matching matrices; the MASt3R backbone uses the GPU version
in ``src/fast_matching.py``, the VGGT backbone this one.

1. ``patch_cross_matrix``: the 16x16-patch matching matrix (rows: view-1 patches, cols: view-2 patches;
   entry = number of matched pixel pairs). 16 px is the MASt3R ViT patch size, so the 32 x 32 patches of a
   512 x 512 image index the decoder tokens 1:1 and the matrix is used directly as the attention mask of the
   feature aggregation (``src/feature_aggregation.py``).
2. ``matching_distance_mask``: for every matched patch pair, view-2 pixels whose 3D point lies within
   ``threshold`` of the view-1 patch's points are marked redundant. The threshold is in the unit of the
   MASt3R points, which are metric (metres): ``dist_threshold`` is 0.03 in configs/base.yaml and 0.07 in the
   published configs/csg_fusion.yaml.
"""
import numpy as np
import torch
from scipy.sparse import csr_matrix

IMG_SIZE = 512
PATCH_SIZE = 16


def patch_cross_matrix(coords1, coords2, H=IMG_SIZE, W=IMG_SIZE, patch_size=PATCH_SIZE):
    """Matching matrix between the patches of the two images (duplicates summed, as csr_matrix does)."""
    n_cols = W // patch_size
    n_patches = (H // patch_size) * n_cols
    coords1, coords2 = np.asarray(coords1), np.asarray(coords2)
    rows = (coords1[:, 1] // patch_size).astype(np.int64) * n_cols + (coords1[:, 0] // patch_size).astype(np.int64)
    cols = (coords2[:, 1] // patch_size).astype(np.int64) * n_cols + (coords2[:, 0] // patch_size).astype(np.int64)
    return csr_matrix((np.ones(len(rows), dtype=np.int64), (rows, cols)), shape=(n_patches, n_patches))


def _patches(point_map, patch_size=PATCH_SIZE):
    """(H, W, 3) -> (n_patches, patch_size*patch_size, 3), row-major patch order."""
    H, W, C = point_map.shape
    p = point_map.reshape(H // patch_size, patch_size, W // patch_size, patch_size, C)
    return p.permute(0, 2, 1, 3, 4).reshape(-1, patch_size * patch_size, C)


@torch.no_grad()
def matching_distance_mask(point_map1, point_map2, cross_mat, threshold=0.10, patch_size=PATCH_SIZE, chunk=256, union=False):
    """Boolean (H, W) mask on point_map1's device: True = view-2 pixel is redundant (within threshold).

    Each view-2 pixel's distance is the minimum over the 256 points of the compared view-1 patch, i.e. the
    distance to its nearest view-1 point in that patch.

    union=False (published model): each view-2 patch is compared with the last matched view-1 patch only, i.e. the
    one with the largest index; the other matched view-1 patches are ignored (``model.filter_union_matches``
    fixes this).
    union=True: a view-2 pixel is redundant if it is within threshold of any matched view-1 patch."""
    H, W, _ = point_map2.shape
    n_cols = W // patch_size
    mask = torch.zeros(H // patch_size * n_cols, patch_size * patch_size, dtype=torch.bool, device=point_map1.device)
    coo = cross_mat.tocoo()
    if coo.nnz:
        if union:
            p1, p2 = coo.row.astype(np.int64), coo.col.astype(np.int64)    # every matched pair (csr has no duplicates)
        else:
            last_row = np.full(cross_mat.shape[1], -1, dtype=np.int64)
            np.maximum.at(last_row, coo.col, coo.row)
            p2 = np.nonzero(last_row >= 0)[0]
            p1 = last_row[p2]
        patches1 = _patches(point_map1.double(), patch_size)
        patches2 = _patches(point_map2.double(), patch_size)
        p1_t = torch.as_tensor(p1, device=mask.device)
        p2_t = torch.as_tensor(p2, device=mask.device)
        for s in range(0, len(p2), chunk):
            c1, c2 = patches1[p1_t[s:s + chunk]], patches2[p2_t[s:s + chunk]]
            d = c2[:, :, None, :] - c1[:, None, :, :]                       # (n, 256 view-2 pts, 256 view-1 pts, 3)
            dist = (d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]).sqrt()
            close = dist.min(dim=2).values < threshold
            if union:
                mask.index_put_((p2_t[s:s + chunk],), close, accumulate=True)  # logical OR over pairs
            else:
                mask[p2_t[s:s + chunk]] = close
    # back to image layout
    return mask.view(H // patch_size, n_cols, patch_size, patch_size).permute(0, 2, 1, 3).reshape(H, W)
