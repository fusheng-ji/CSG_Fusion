"""SSIM metric (scikit-image), per image or as a per-pixel map for the masked mean of the evaluation."""
import torch
from skimage.metrics import structural_similarity
import numpy as np


@torch.no_grad()
def compute_ssim(ground_truth, predicted, full=True):
    """(N, 3, H, W) images in [0, 1] -> (N,) SSIM values, or with ``full`` the (N, 3, H, W) SSIM maps (which
    the caller averages over the loss mask). Computed on the CPU with scikit-image."""
    # The arguments to `structural_similarity` have been chosen to match
    # PixelSplat (apart from `full = full`)
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
            full=full,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    if full:
        ssim = [spatial for _, spatial in ssim]
    ssim = np.array(ssim)
    ssim = torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)
    assert not torch.isnan(ssim).any(), "SSIM has NaNs"
    return ssim
