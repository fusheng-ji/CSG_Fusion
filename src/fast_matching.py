"""Matching-based Gaussian Filter (paper Sec. 4.1) on the GPU, used by the CSG-Fusion head (``src/csg_gaussian_head.py``).

* ``reciprocal_matches``: reciprocal (mutual) nearest-neighbour matches between the two descriptor maps, as MASt3R's
  ``fast_reciprocal_NNs``, with the bookkeeping kept on the GPU: only the row count of each nearest-neighbour query is
  read back. Seeds are taken every 8 pixels, and matches within 3 pixels of the image border are dropped, as in
  MASt3R's matching example.
* ``patch_counts``: the 1024 x 1024 patch matching matrix as a dense tensor of match counts (rows: view-1 patches,
  cols: view-2 patches), which the feature aggregation uses as its attention mask.
* ``distance_mask``: the view-2 pixels whose 3D point lies within the threshold of their matched view-1 patch. The
  distance of every view-2 patch is computed (unmatched ones are masked out afterwards), so there is no host sync.
"""
import math

import torch

from src.matching_filter import IMG_SIZE, PATCH_SIZE, _patches

# The fast_reciprocal_NNs call of MASt3R's matching example: seeds every 8 px, dot-product NN in blocks of 2**13, at most 10 back-and-forth iterations (the
# function's own limit for an integer seed stride), matches within 3 px of the image border dropped.
SUBSAMPLE = 8
BLOCK_SIZE = 2 ** 13
MAX_ITER = 10
BORDER = 3


@torch.no_grad()
def _blocked_dot_nn(A, B, block_size=BLOCK_SIZE, group=8):
    """``mast3r.fast_nn.blocked_dot_nn`` without the final ``.cpu().numpy()`` (same ops, same result)."""
    device = A.device
    n_blocks = math.ceil(B.shape[0] / block_size)
    pad = n_blocks * block_size - B.shape[0]
    B_blocks = torch.cat([B, B.new_zeros(pad, B.shape[1])]).view(n_blocks, block_size, -1) if pad else B.view(n_blocks, block_size, -1)
    nn_A = torch.empty(A.shape[0], dtype=torch.int64, device=device)
    for i in range(math.ceil(A.shape[0] / block_size)):
        A_i = A[i * block_size:(i + 1) * block_size]
        sims, args = [], []
        for g in range(0, n_blocks, group):
            sim = torch.matmul(A_i[None], B_blocks[g:g + group].transpose(1, 2))
            if pad and g + group >= n_blocks:
                sim[-1, :, block_size - pad:] = -float('inf')
            s, a = sim.max(dim=2)
            sims.append(s)
            args.append(a + torch.arange(g, g + s.shape[0], device=device)[:, None] * block_size)
        best = torch.cat(sims).max(dim=0).indices
        nn_A[i * block_size:(i + 1) * block_size] = torch.cat(args).gather(0, best[None])[0]
    return nn_A


@torch.no_grad()
def _query(queries, db, block_size=BLOCK_SIZE):
    """``cdistMatcher(db).query(queries, dist='dot', block_size=block_size)``, result on the GPU."""
    if len(queries) * len(db) > block_size ** 2:
        return _blocked_dot_nn(queries, db, block_size=block_size)
    # bruteforce_reciprocal_nns(dist='dot') with a single block: nn_A = argmax of A @ B.T (nn_B is unused)
    return torch.max(queries @ db.T, dim=1)[1]


@torch.no_grad()
def reciprocal_matches(desc1, desc2, subsample=SUBSAMPLE, max_iter=MAX_ITER, border=BORDER):
    """(H, W, D) descriptor maps -> (m1, m2, valid): (n, 2) int64 xy pixels in view 1 / view 2 and an (n,)
    bool mask. ``m1[valid], m2[valid]`` are the matches of one batch item."""
    H1, W1, D = desc1.shape
    H2, W2, _ = desc2.shape
    device = desc1.device
    pts1, pts2 = desc1.reshape(-1, D), desc2.reshape(-1, D)
    ys = torch.arange(subsample // 2, H1, subsample, device=device)
    xs = torch.arange(subsample // 2, W1, subsample, device=device)
    xy1 = (xs[None, :] + W1 * ys[:, None]).reshape(-1)              # already sorted and unique
    xy2 = torch.full_like(xy1, -1)
    old_xy1, old_xy2 = xy1.clone(), xy2.clone()
    notyet = torch.ones(len(xy1), dtype=torch.bool, device=device)

    for niter in range(1, max_iter + 1):
        idx = notyet.nonzero()[:, 0]                                 # host sync: the query size
        if len(idx) == 0:                                            # `while notyet.any()`
            break
        xy2[idx] = _query(pts1[xy1[idx]], pts2)
        notyet &= old_xy2 != xy2
        idx = notyet.nonzero()[:, 0]
        if len(idx):
            xy1[idx] = _query(pts2[xy2[idx]], pts1)
        notyet &= old_xy1 != xy1
        if niter >= max_iter:
            break
        old_xy2.copy_(xy2)
        old_xy1.copy_(xy1)

    # merge_corres: keep converged, valid pairs; unique on (xy1, xy2), sorted by xy1 then xy2. Done with a
    # sort and a fixed-size validity mask instead of torch.unique / boolean indexing (no host sync).
    keep = ~notyet & (xy1 >= 0) & (xy2 >= 0)
    n2 = H2 * W2
    key = torch.where(keep, xy1 * n2 + xy2, H1 * W1 * n2).sort().values      # dropped pairs sort last
    first = torch.ones_like(keep)
    first[1:] = key[1:] != key[:-1]
    i1, i2 = key // n2, key % n2
    m1 = torch.stack((i1 % W1, i1 // W1), dim=1)                     # (x, y)
    m2 = torch.stack((i2 % W2, i2 // W2), dim=1)
    inside = lambda m, H, W: (m[:, 0] >= border) & (m[:, 0] < W - border) & (m[:, 1] >= border) & (m[:, 1] < H - border)
    valid = first & (key < H1 * W1 * n2) & inside(m1, IMG_SIZE, IMG_SIZE) & inside(m2, IMG_SIZE, IMG_SIZE)
    return m1, m2, valid


def patch_counts(m1, m2, valid, H=IMG_SIZE, W=IMG_SIZE, patch_size=PATCH_SIZE):
    """Dense (n_patches, n_patches) int64 matrix of match counts."""
    n_cols = W // patch_size
    n_patches = (H // patch_size) * n_cols
    rows = (m1[:, 1] // patch_size) * n_cols + m1[:, 0] // patch_size
    cols = (m2[:, 1] // patch_size) * n_cols + m2[:, 0] // patch_size
    counts = torch.zeros(n_patches * n_patches, dtype=torch.int64, device=m1.device)
    index = torch.where(valid, rows * n_patches + cols, 0)          # dropped entries (weight 0) may be out of range
    return counts.index_add_(0, index, valid.long()).view(n_patches, n_patches)


@torch.no_grad()
def distance_mask(point_map1, point_map2, counts, threshold=0.10, patch_size=PATCH_SIZE, chunk=256, union=False):
    """(H, W) bool, True = view-2 pixel is redundant (within threshold of its matched view-1 patch).

    union=True (model.filter_union_matches): compare each view-2 patch with every matched view-1 patch."""
    H, W, _ = point_map2.shape
    n_cols = W // patch_size
    n_patches = counts.shape[1]
    if union:
        p1_all, p2_all = torch.nonzero(counts > 0, as_tuple=True)
        patches1 = _patches(point_map1.double(), patch_size)
        patches2 = _patches(point_map2.double(), patch_size)
        hit = torch.zeros(n_patches, patch_size * patch_size, dtype=torch.int32, device=point_map1.device)
        for s in range(0, len(p2_all), chunk):
            c1, c2 = patches1[p1_all[s:s + chunk]], patches2[p2_all[s:s + chunk]]
            d = c2[:, :, None, :] - c1[:, None, :, :]
            dist = (d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]).sqrt()
            hit.index_add_(0, p2_all[s:s + chunk], (dist.min(dim=2).values < threshold).to(torch.int32))
        return (hit > 0).view(H // patch_size, n_cols, patch_size, patch_size).permute(0, 2, 1, 3).reshape(H, W)
    # the largest matching view-1 patch decides each view-2 patch (-1: unmatched)
    row_idx = torch.arange(n_patches, device=counts.device)[:, None].expand_as(counts)
    last_row = torch.where(counts > 0, row_idx, -1).amax(dim=0)
    matched = last_row >= 0
    # Unmatched view-2 patches are compared with view-1 patch 0 and then masked out with `matched`: every
    # chunk has a fixed size, so no host sync is needed to select the matched patches first.
    p1 = last_row.clamp_min(0)
    patches1 = _patches(point_map1.double(), patch_size)
    patches2 = _patches(point_map2.double(), patch_size)
    mask = torch.empty(n_patches, patch_size * patch_size, dtype=torch.bool, device=point_map1.device)
    for s in range(0, n_patches, chunk):
        c1, c2 = patches1[p1[s:s + chunk]], patches2[s:s + chunk]
        d = c2[:, :, None, :] - c1[:, None, :, :]
        dist = (d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]).sqrt()
        mask[s:s + chunk] = (dist.min(dim=2).values < threshold) & matched[s:s + chunk, None]
    return mask.view(H // patch_size, n_cols, patch_size, patch_size).permute(0, 2, 1, 3).reshape(H, W)
