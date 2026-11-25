"""Training-free cross-view Gaussian fusion: a plug-in version of the Matching-based Gaussian Filter.

CSG-Fusion drops view-2 Gaussians whose 3D position duplicates view-1 geometry (``src/matching_filter.py``,
with MASt3R descriptor matches). This module applies the same idea to *any* feed-forward method that predicts
Gaussians from two views, without retraining and without a matching head. The correspondence comes from
geometry alone:

* ``depth``: project every view-2 Gaussian centre into view 1; it is redundant if it lands inside view 1 and
  its depth agrees with view 1's surface there, ``|z2 - Z1| < tau * Z1``. ``Z1`` is a z-buffer of the view-1
  Gaussian centres (nearest centre per pixel, opacity above ``min_opacity``), so the test only needs the
  Gaussian sets and view 1's camera -- not the method's pixel layout.
* ``merge``: as ``depth``, but instead of only dropping a redundant view-2 Gaussian, fold its opacity into
  the view-1 Gaussian that owns that z-buffer pixel by alpha compositing, ``a1 <- 1 - (1 - a1) * prod(1 - a2)``.
  Feed-forward methods trained without fusion split a surface's opacity over both views' Gaussians, so a
  plain drop leaves it semi-transparent.
* ``random``: control. Drop the same number of view-2 Gaussians uniformly at random.

All functions work on one batch item (the harness evaluates with batch size 1). Used by the baseline
evaluation (``scripts/eval/baselines``), not by the CSG-Fusion model itself.
"""
import torch


@torch.no_grad()
def zbuffer(means_cam, K, hw, min_depth=1e-6, return_index=False):
    """Nearest depth per pixel of points ``means_cam`` (N, 3) in camera space; ``K`` (3, 3) in pixels.

    Returns ``(H, W)`` with +inf where no point projects (and, with ``return_index``, the (H, W) index of the
    nearest point, -1 where none)."""
    H, W = hw
    z = means_cam[:, 2]
    uv = (means_cam @ K.T)[:, :2] / z.clamp_min(min_depth)[:, None]
    # floor: each point goes to the single pixel whose square [u, u+1) x [v, v+1) contains its projection
    # (no splat footprint), and each pixel keeps the nearest such point (a nearest-point z-buffer)
    u, v = uv[:, 0].floor().long(), uv[:, 1].floor().long()
    ok = (z > min_depth) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    buf = torch.full((H * W,), float('inf'), device=means_cam.device, dtype=means_cam.dtype)
    pix = (v * W + u)[ok]
    buf.scatter_reduce_(0, pix, z[ok], reduce='amin')
    if not return_index:
        return buf.view(H, W)
    idx = torch.full((H * W,), -1, device=means_cam.device, dtype=torch.long)
    ids = torch.nonzero(ok).squeeze(1)
    nearest = z[ok] == buf[pix]
    # several points at exactly the nearest depth: the one with the highest index wins
    idx.scatter_reduce_(0, pix[nearest], ids[nearest], reduce='amax')
    return buf.view(H, W), idx.view(H, W)


@torch.no_grad()
def redundant_mask(means2, means1, c2w1, K1, hw, tau, opacities1=None, min_opacity=0.0, border=0,
                   return_partner=False):
    """Boolean (N2,): True = view-2 Gaussian duplicates view-1 geometry (with ``return_partner``, also the
    (N2,) index into ``means1`` of the view-1 Gaussian it duplicates, -1 where not redundant).

    means1 / means2: (N, 3) world-space centres; c2w1: (4, 4) camera-to-world of view 1 (OpenCV);
    K1: (3, 3) pixel intrinsics for an image of size ``hw``; tau: relative depth tolerance."""
    w2c = torch.linalg.inv(c2w1.double())
    def to_cam(x):
        return (x.double() @ w2c[:3, :3].T + w2c[:3, 3])
    K = K1.double()
    sel = torch.arange(means1.shape[0], device=means1.device)
    if opacities1 is not None and min_opacity > 0:
        sel = sel[opacities1.reshape(-1) > min_opacity]
    Z1, I1 = zbuffer(to_cam(means1[sel]), K, hw, return_index=True)
    p2 = to_cam(means2)
    H, W = hw
    z2 = p2[:, 2]
    uv = (p2 @ K.T)[:, :2] / z2.clamp_min(1e-6)[:, None]
    u, v = uv[:, 0].floor().long(), uv[:, 1].floor().long()
    inside = (z2 > 1e-6) & (u >= border) & (u < W - border) & (v >= border) & (v < H - border)
    z1 = torch.full_like(z2, float('inf'))
    z1[inside] = Z1[v[inside], u[inside]]
    red = inside & torch.isfinite(z1) & ((z2 - z1).abs() < tau * z1)
    if not return_partner:
        return red
    partner = torch.full_like(u, -1)
    partner[red] = sel[I1[v[red], u[red]]]
    return red, partner


@torch.no_grad()
def merge_opacities(opacities1, opacities2, red, partner):
    """View-1 opacities after compositing the redundant view-2 Gaussians onto their partners."""
    a1, a2 = opacities1.reshape(-1).double(), opacities2.reshape(-1).double()
    # prod(1 - a2) over the Gaussians merged into one partner, accumulated as a sum of log1p(-a2) (index_add_
    # cannot multiply); a2 is clamped below 1 so the log stays finite
    log_t = torch.zeros_like(a1).index_add_(0, partner[red], torch.log1p(-a2[red].clamp(max=1 - 1e-6)))
    return (1 - (1 - a1) * log_t.exp()).to(opacities1.dtype).view_as(opacities1)


def random_mask(n2, n_drop, generator=None, device='cpu'):
    """Control: ``n_drop`` of ``n2`` view-2 Gaussians, uniformly at random."""
    m = torch.zeros(n2, dtype=torch.bool, device=device)
    if n_drop > 0:
        m[torch.randperm(n2, generator=generator, device='cpu')[:n_drop].to(device)] = True
    return m
