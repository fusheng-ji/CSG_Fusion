# Adapted from pixelSplat (https://github.com/dcharatan/pixelsplat), MIT License,
# Copyright (c) 2023 David Charatan, Sizhe Li, Andrea Tagliasacchi and Vincent Sitzmann. Modified for CSG-Fusion.
"""Renderer: rasterises the predicted Gaussians into the target (and optionally context) views.

The rendered set is every view-1 Gaussian plus the view-2 Gaussians kept by the Matching-based Gaussian Filter
(``pred2['mask'] == 1``), in view 1's camera frame (all camera poses are expressed relative to context view 1).
Rasterisation is pixelSplat's ``render_cuda`` (``cuda_splatting.py``).
"""
import torch
from einops import rearrange, repeat
import cv2
import numpy as np

from .cuda_splatting import render_cuda
from utils.geometry import normalize_intrinsics


def _get_pose_from_pnp(pts3d, pixels, K, device):
    """Camera-to-world pose from 2D-3D correspondences (EPnP + RANSAC); identity if PnP fails."""
    pts3d_np = pts3d.cpu().numpy()
    pixels_np = pixels.cpu().numpy().astype(np.float32)
    K_np = K.cpu().numpy()
    try:
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            pts3d_np, pixels_np, K_np, None, iterationsCount=1000, reprojectionError=4.0, flags=cv2.SOLVEPNP_EPNP
        )
    except cv2.error:
        success = False
    if not success:
        return torch.eye(4, device=device, dtype=torch.float32)
    R, _ = cv2.Rodrigues(rvec)
    w2c = torch.eye(4, device=device, dtype=torch.float32)
    w2c[:3, :3] = torch.from_numpy(R).float().to(device)
    w2c[:3, 3] = torch.from_numpy(tvec).float().to(device).squeeze(-1)
    return torch.inverse(w2c)

class DecoderSplattingCUDA(torch.nn.Module):
    """Gaussian-splatting decoder: forward(batch, pred1, pred2, render_context) returns the target
    renders (B, V, 3, H, W) and the context renders (or None)."""

    def __init__(self, background_color):
        super().__init__()
        # model.pad_detach_fix: only the padding dummies are cut from the graph, so every batch item's
        # view-2 Gaussians get gradients (False, as in the published model: only those of the item(s) with the most kept Gaussians)
        self.pad_detach_fix = False
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, batch, pred1, pred2, render_context=True):
        """Two-view rendering: all view-1 Gaussians + the view-2 Gaussians kept by ``pred2['mask']``."""
        return self.forward_multi(batch, [pred1, pred2], render_context)

    def forward_multi(self, batch, preds, render_context=True):
        """N-view rendering (src/multiview.py): every Gaussian of ``preds[0]`` (the reference view, key
        ``means``) + the Gaussians of ``preds[1:]`` (keys ``means_in_other_view``, all in the reference frame)
        kept by their ``mask``. With two views this is exactly ``forward``."""
        pred1, pred2 = preds[0], preds[1]   # pred2: the view used by the DTU alignment below
        image_shape = batch['target'][0]['original_img'].shape[-2:]  # (H, W); training target views carry no 'img'
        base_pose = batch['context'][0]['camera_pose'] # [b, 4, 4]
        inv_base_pose = torch.inverse(base_pose)
        intrinsics = torch.stack([target_view['camera_intrinsics'] for target_view in batch['target']], dim=1)
        intrinsics = normalize_intrinsics(intrinsics, image_shape)[..., :3, :3]
        target_extrinsics = torch.stack([target_view['camera_pose'] for target_view in batch['target']], dim=1)
        extrinsics = inv_base_pose[:, None, :, :] @ target_extrinsics
        context_intrinsics = torch.stack([context_view['camera_intrinsics'] for context_view in batch['context']], dim=1)
        context_intrinsics = normalize_intrinsics(context_intrinsics, image_shape)[..., :3, :3]
        context_extrinsics = torch.stack([context_view['camera_pose'] for context_view in batch['context']], dim=1)
        context_extrinsics = inv_base_pose[:, None, :, :] @ context_extrinsics

        # DTU only (assumes batch size 1): compute and apply the alignment transform
        # to align the predicted coordinate system with the ground truth relative system. The predicted pose
        # of context view 2 comes from PnP on its predicted points (in view 1's frame) and its pixel grid.
        if 'dtu' in batch['context'][0].get('dataset', ''):
            with torch.no_grad():
                # Get predicted relative pose for the second camera
                p_in_frame1 = pred2['pts3d_in_other_view'].squeeze(0).reshape(-1, 3)
                H, W = pred2['pts3d_in_other_view'].shape[1:3]
                pixels_in_frame2 = torch.from_numpy(np.mgrid[0:W, 0:H].T.reshape(-1, 2)).float().to(p_in_frame1.device)
                K2 = batch['context'][1]['camera_intrinsics'].squeeze(0)
                cam2w_pred_2 = _get_pose_from_pnp(p_in_frame1, pixels_in_frame2, K2, device=p_in_frame1.device)

                # Get ground truth relative pose for the second camera
                cam2w_gt_relative_2 = context_extrinsics.squeeze(0)[1]

                # Align the second camera poses to find the transformation between coordinate systems
                # T_gt_from_pred maps a pose from the predicted space to the GT relative space
                T_gt_from_pred = cam2w_gt_relative_2 @ torch.inverse(cam2w_pred_2)

                # The inverse maps GT poses into the model's predicted space for rendering
                T_model_from_gt = torch.inverse(T_gt_from_pred)

                # Apply the alignment to the already-relative poses.
                # ALL poses (target and context) must be transformed to the model's predicted coordinate system
                extrinsics = T_model_from_gt[None, None, :, :] @ extrinsics

                # For context views, transform both cameras to the predicted space
                context_extrinsics = T_model_from_gt[None, None, :, :] @ context_extrinsics

        # ---- Render the target views from view-1 Gaussians + the view-2 Gaussians kept by the filter ----
        b, v, _, _ = extrinsics.shape
        # fixed near 0.1 / far 1000 for every view (scene units, metres for MASt3R points): render_cuda
        # rescales the scene by 1 / near and builds the projection matrix from them
        near = torch.full((b, v), 0.1, device=extrinsics.device)
        far = torch.full((b, v), 1000.0, device=extrinsics.device)

        # The non-reference views are flattened and concatenated along the Gaussian axis (pixel order, view
        # by view); with two views the concatenation is just view 2.
        others = preds[1:]
        flat = lambda key, pattern: torch.cat([rearrange(p[key], pattern) for p in others], dim=1)
        mask_flat = torch.cat([p['mask'].squeeze(-1).bool().reshape(b, -1) for p in others], dim=1)  # B x (n*H*W)
        means1_flat = rearrange(pred1['means'], "b h w xyz -> b (h w) xyz")
        means2_flat = flat('means_in_other_view', "b h w xyz -> b (h w) xyz")
        cov1_flat = rearrange(pred1["covariances"], "b h w i j -> b (h w) i j")
        cov2_flat = flat("covariances", "b h w i j -> b (h w) i j")
        harm1_flat = rearrange(pred1["sh"], "b h w c d_sh -> b (h w) c d_sh")
        harm2_flat = flat("sh", "b h w c d_sh -> b (h w) c d_sh")
        opa1_flat = rearrange(pred1["opacities"], "b h w 1 -> b (h w) 1")
        opa2_flat = flat("opacities", "b h w 1 -> b (h w) 1")
        # Keep the view-2 Gaussians that survived the matching-based filter (in pixel order), padded
        # to the largest count in the batch, because the batched rasterizer input needs the same number of
        # Gaussians per item. The dummies render nothing because their opacity is 0 (alpha 0 adds no colour
        # and no occlusion; the bundled rasterizer, third_party/diff-gaussian-rasterization, also skips
        # zero-opacity Gaussians in its preprocessing). The mean (0, 0, 1e6), far away along +z, covariance
        # 1e10 * I and SH 0 are placeholder values (the rasterizer culls by a near-plane test only, not by
        # `far`). A batch item that needs padding has its view-2 Gaussians detached (model.pad_detach_fix keeps
        # their gradients).
        counts = mask_flat.sum(dim=1)
        max_g2 = int(counts.max())
        order = torch.argsort((~mask_flat).to(torch.uint8), dim=1, stable=True)[:, :max_g2]  # valid first
        in_range = torch.arange(max_g2, device=mask_flat.device)[None, :] < counts[:, None]  # B x max_g2
        keep_grad = counts == max_g2

        def select(x, pad_value):
            idx = order.view(b, max_g2, *([1] * (x.dim() - 2))).expand(b, max_g2, *x.shape[2:])
            x = x.gather(1, idx)
            if not self.pad_detach_fix:  # detach every Gaussian of an item that needs padding
                x = torch.where(keep_grad.view(b, *([1] * (x.dim() - 1))), x, x.detach())
            return torch.where(in_range.view(b, max_g2, *([1] * (x.dim() - 2))), x, pad_value)

        dummy_mean = torch.zeros(3, device=means2_flat.device, dtype=means2_flat.dtype)
        dummy_mean[2] = 1e6
        dummy_cov = torch.eye(3, device=cov2_flat.device, dtype=cov2_flat.dtype) * 1e10
        means2_padded = select(means2_flat, dummy_mean)  # B x max_g2 x xyz
        cov2_padded = select(cov2_flat, dummy_cov)
        harm2_padded = select(harm2_flat, 0.0)
        opa2_padded = select(opa2_flat, 0.0)  # B x max_g2 x 1

        # === concat view1 + filtered+pad view2 ===
        means_cat = torch.cat([means1_flat, means2_padded], dim=1)  # B x (g1 + g2) x xyz
        cov_cat = torch.cat([cov1_flat, cov2_padded], dim=1)
        harm_cat = torch.cat([harm1_flat, harm2_padded], dim=1)
        opa_cat = torch.cat([opa1_flat, opa2_padded], dim=1)

        # === repeat for per-view ===
        color = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(means_cat, "b g xyz -> (b v) g xyz", v=v),
            repeat(cov_cat, "b g i j -> (b v) g i j", v=v),
            repeat(harm_cat, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(opa_cat.squeeze(-1), "b g -> (b v) g", v=v),
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        if not render_context:  # the caller does not use the context renders (no context loss)
            return color, None

        b, v, _, _ = context_extrinsics.shape
        near = torch.full((b, v), 0.1, device=context_extrinsics.device)
        far = torch.full((b, v), 1000.0, device=context_extrinsics.device)

        context_color = render_cuda(
            rearrange(context_extrinsics, "b v i j -> (b v) i j"),
            rearrange(context_intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(means_cat, "b g xyz -> (b v) g xyz", v=v),
            repeat(cov_cat, "b g i j -> (b v) g i j", v=v),
            repeat(harm_cat, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(opa_cat.squeeze(-1), "b g -> (b v) g", v=v),
        )
        context_color = rearrange(context_color, "(b v) c h w -> b v c h w", b=b, v=v)

        return color, context_color
