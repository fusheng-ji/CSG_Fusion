"""CSG-Fusion on a VGGT backbone (``model.backbone: vggt``).

The same method as the MASt3R backbone (``src/csg_gaussian_head.py``), with every MASt3R part
replaced by its VGGT counterpart:

    MASt3R (external)                         VGGT
    frozen ViT encoder + decoder        ->    frozen VGGT-1B aggregator (layers 4 / 11 / 17 / 23 cached)
    pts3d head (fine-tuned, metric)     ->    VGGT point head (fine-tuned to metric with the same ConfLoss)
                                              x a learned global scale (VGGT points are scale-normalised)
    24-d matching descriptor `desc`     ->    VGGT track-head feature maps (frozen), L2-normalised
    Matching-based Gaussian Filter      ->    unchanged (src/matching_filter.py at 512 / 16-px patches)
    Mask-based Feature Aggregation      ->    unchanged module (src/feature_aggregation.FeatureFusion) on
                                              VGGT patch tokens (37 x 37 grid, 14-px patches at 518)
    two Gaussian DPT heads on decoder   ->    two VGGT DPT heads (feature-only) on aggregator tokens
    tokens                                    + a small conv head with the last-layer initialisation of
                                              src/csg_gaussian_head.py

The data pipeline stays at 512 x 512: images go to VGGT at 518 x 518 and every per-pixel output (points,
confidence, descriptors, Gaussian parameters) is resized back to 512, so the filter, the decoder, the losses
and the evaluation are exactly those of the MASt3R backbone.

perf.matching_subsample and perf.checkpoint_gaussian_dpt apply to the MASt3R backbone only.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

_VGGT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party', 'vggt')
if _VGGT not in sys.path:
    sys.path.append(_VGGT)

from vggt.heads.dpt_head import DPTHead  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402

from src.matching_filter import matching_distance_mask, patch_cross_matrix  # noqa: E402
from src.feature_aggregation import FeatureFusion  # noqa: E402
from mast3r.fast_nn import fast_reciprocal_NNs  # noqa: E402

VGGT_RES = 518
VGGT_PATCH = 14
TOKEN_LAYERS = (4, 11, 17, 23)


def _resize_maps(x, size):
    """(B, S, C, h, w) -> (B, S, C, H, W), bilinear."""
    b, s = x.shape[:2]
    return F.interpolate(x.flatten(0, 1), size=size, mode='bilinear', align_corners=False).unflatten(0, (b, s))


class VGGTBackbone(nn.Module):
    """Frozen VGGT aggregator + track features; trainable point head and global metric scale."""

    def __init__(self, pretrained='facebook/VGGT-1B', init_log_scale=0.0, bf16=True):
        super().__init__()
        vggt = VGGT.from_pretrained(pretrained) if pretrained else VGGT()
        self.aggregator = vggt.aggregator
        self.point_head = vggt.point_head
        self.track_features = vggt.track_head.feature_extractor
        del vggt
        self.aggregator.requires_grad_(False)
        self.track_features.requires_grad_(False)
        self.log_scale = nn.Parameter(torch.tensor(float(init_log_scale)))
        self.bf16 = bf16

    def train(self, mode=True):
        super().train(mode)
        self.aggregator.eval()            # frozen parts stay in eval mode
        self.track_features.eval()
        return self

    def forward(self, view1, view2, out_hw):
        res, tokens, patch_start, imgs = self.forward_views([view1, view2], out_hw)
        return res[0], res[1], tokens, patch_start, imgs

    def forward_views(self, views, out_hw):
        """Any number of views in one aggregator pass (VGGT is a multi-frame model; frame 0 is the reference
        whose camera frame all points are in). Returns one result dict per view."""
        imgs = torch.stack([v['original_img'] for v in views], dim=1)                   # (B, S, 3, H, W) in [0, 1]
        imgs = _resize_maps(imgs, (VGGT_RES, VGGT_RES)).clamp(0, 1)
        # Frozen parts: no autograd, optionally under bf16 autocast. The point head, the trainable part (fine-tuned
        # by the MASt3R confidence loss when it is enabled), runs with autograd and in fp32 (autocast disabled,
        # tokens cast back to fp32).
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
            tokens, patch_start = self.aggregator(imgs)
            desc = self.track_features(tokens, imgs, patch_start)                        # (B, 2, 128, 259, 259)
        tokens = [t.float() if t is not None else None for t in tokens]
        with torch.autocast('cuda', enabled=False):
            pts, conf = self.point_head(tokens, images=imgs, patch_start_idx=patch_start)  # (B,2,518,518,3), (B,2,518,518)
        pts = pts * self.log_scale.exp()
        pts = _resize_maps(pts.permute(0, 1, 4, 2, 3), out_hw).permute(0, 1, 3, 4, 2)
        conf = _resize_maps(conf[:, :, None], out_hw)[:, :, 0]
        # L2-normalised track features: the 'dot' nearest-neighbour search of the matching is then cosine similarity
        desc = F.normalize(_resize_maps(desc.float(), out_hw), dim=2).permute(0, 1, 3, 4, 2)
        res = [{'pts3d': pts[:, s], 'conf': conf[:, s], 'desc': desc[:, s]} for s in range(len(views))]
        return res, tokens, patch_start, imgs


class VGGTGaussianHead(nn.Module):
    """The CSG-Fusion GaussianHead (filter + feature aggregation + two Gaussian DPT heads) on VGGT tokens.

    ``dist_threshold`` defaults to 0.07, the value of the published recipe (configs/csg_fusion_vggt.yaml).
    The outputs (``pred['scales']``, ``rotations``, ``sh``, ``opacities``, ``means``, ``pred2['mask']``,
    ``pred2['cross_mat']``) have the same layout as those of ``src.csg_gaussian_head.GaussianHead``."""

    def __init__(self, use_attention=True, attention_neighbor=0, use_offsets=True, sh_degree=1,
                 dist_threshold=0.07, dim_in=2048, features=256):
        super().__init__()
        self.use_attention = use_attention
        self.use_offsets = use_offsets
        self.sh_degree = sh_degree
        self.dist_threshold = dist_threshold
        self.filter_union_matches = False  # model.filter_union_matches
        self.splits = [3, 3, 4, 3 * sh_degree, 1]
        n_out = sum(self.splits)
        self.dpt = nn.ModuleList([DPTHead(dim_in=dim_in, patch_size=VGGT_PATCH, features=features, feature_only=True,
                                          intermediate_layer_idx=list(TOKEN_LAYERS)) for _ in range(2)])
        self.out = nn.ModuleList([nn.Sequential(nn.Conv2d(features, 64, 3, padding=1), nn.ReLU(inplace=True),
                                                nn.Conv2d(64, n_out, 1)) for _ in range(2)])
        for head in self.out:                     # src/csg_gaussian_head.py initialisation of the last layer
            last, start = head[-1], 0
            for n, s, b in [(3, 0.001, 0.001), (3, 0.00003, -7.0), (4, 1.0, 0.0), (3 * sh_degree, 1.0, 0.0), (1, 1.0, -2.0)]:
                nn.init.xavier_uniform_(last.weight[start:start + n], s)
                nn.init.constant_(last.bias[start:start + n], b)
                start += n
        self.grid = VGGT_RES // VGGT_PATCH                                                 # 37
        if use_attention:
            d = dim_in * len(TOKEN_LAYERS)
            self.fusion_module = FeatureFusion(input_dim=d, hidden_dim=512, output_dim=d, num_heads=8, dropout=0.1,
                                               neighborhood_size=attention_neighbor, grid_size=self.grid)

    @staticmethod
    def reg_offsets(xyz, shift=6.0):
        """Small offsets at initialisation: the formula of ``GaussianHead.reg_offsets``."""
        d = xyz.norm(dim=-1, keepdim=True)
        return xyz / d.clip(min=1e-8) * (torch.exp(d - shift) - math.exp(-shift))

    def _matches(self, desc1, desc2, hw):
        """Reciprocal matches of VGGT descriptors, for any image size (seed stride 8, 3-px border)."""
        H, W = hw
        out1, out2 = [], []
        for i in range(desc1.shape[0]):
            m1, m2 = fast_reciprocal_NNs(desc1[i].detach(), desc2[i].detach(), subsample_or_initxy1=8,
                                         device='cuda', dist='dot', block_size=2 ** 13)
            ok = ((m1[:, 0] >= 3) & (m1[:, 0] < W - 3) & (m1[:, 1] >= 3) & (m1[:, 1] < H - 3) &
                  (m2[:, 0] >= 3) & (m2[:, 0] < W - 3) & (m2[:, 1] >= 3) & (m2[:, 1] < H - 3))
            out1.append(m1[ok])
            out2.append(m2[ok])
        return out1, out2

    def _ref_filter(self, pred_ref, pred_k, hw):
        """Matching-based Gaussian Filter of view k against the reference view: (keep mask (B, H, W, 1), 1 = keep;
        the per-item CSR patch matrices at 512 / 16 px; the same matrices on VGGT's 37 x 37 token grid; the number
        of matches per item)."""
        H, W = hw
        m1s, m2s = self._matches(pred_ref['desc'], pred_k['desc'], (H, W))
        masks, cross_mats, token_mats = [], [], []
        to_tok = VGGT_RES / W
        for i in range(len(m1s)):
            cm = patch_cross_matrix(m1s[i], m2s[i], H, W)                                  # 32 x 32 patches at 512
            cross_mats.append(cm)
            red = matching_distance_mask(pred_ref['pts3d'][i].detach(), pred_k['pts3d'][i].detach(), cm,
                                         threshold=self.dist_threshold, union=self.filter_union_matches)
            masks.append(~red)                                                             # 1 = keep
            # The feature aggregation needs the matching matrix on VGGT's 37 x 37 token grid: match pixels are
            # rescaled by 518 / 512 and floored to 14-px token patches. The filter mask above stays on the
            # 512 / 16-px grid.
            token_mats.append(patch_cross_matrix(np.floor(np.asarray(m1s[i]) * to_tok), np.floor(np.asarray(m2s[i]) * to_tok),
                                                 VGGT_RES, VGGT_RES, VGGT_PATCH))      # 37 x 37 token grid
        mask = torch.stack(masks).float().unsqueeze(3).detach()
        return mask, cross_mats, token_mats, [len(m) for m in m1s]

    def _gaussian_params(self, pred, feat, hw, keep=None):
        """Conv head output (B, C, 518, 518) -> the Gaussian parameters of ``pred`` at hw; ``keep``: filter mask."""
        g = F.interpolate(feat, size=hw, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        offset, scales, rot, sh, opa = torch.split(g, self.splits, dim=-1)
        pred['scales'] = torch.exp(scales)
        pred['rotations'] = rot / (rot.norm(dim=-1, keepdim=True) + 1e-8)
        pred['sh'] = rearrange(sh, '... (xyz d_sh) -> ... xyz d_sh', xyz=3)
        pred['opacities'] = torch.sigmoid(opa)
        if keep is not None:
            pred['opacities'] = torch.where(keep == 0, 0.0, pred['opacities'])
        pts = pred['pts3d'].detach()
        pred['means'] = pts + self.reg_offsets(offset) if self.use_offsets else pts

    def forward(self, pred1, pred2, tokens, patch_start, imgs):
        B, H, W, _ = pred1['pts3d'].shape
        mask, cross_mats, token_mats, _ = self._ref_filter(pred1, pred2, (H, W))
        pred2['mask'], pred2['cross_mat'] = mask, cross_mats

        tok1 = [t[:, 0:1] if t is not None else None for t in tokens]                      # (B, 1, P, 2048) per view
        tok2 = [t[:, 1:2] if t is not None else None for t in tokens]
        if self.use_attention:
            p = patch_start
            f1 = torch.cat([tok1[l][:, 0, p:] for l in TOKEN_LAYERS], dim=-1)               # (B, 1369, 8192)
            f2 = torch.cat([tok2[l][:, 0, p:] for l in TOKEN_LAYERS], dim=-1)
            fused = torch.split(self.fusion_module(f1, f2, token_mats), tokens[TOKEN_LAYERS[0]].shape[-1], dim=-1)
            tok1 = list(tok1)
            for l, f in zip(TOKEN_LAYERS, fused):
                tok1[l] = torch.cat([tok1[l][:, :, :p], f[:, None]], dim=2)

        for v, (pred, tok, keep) in enumerate(((pred1, tok1, None), (pred2, tok2, mask))):
            feat = self.dpt[v](tok, imgs[:, v:v + 1], patch_start)[:, 0]                   # (B, 256, 518, 518)
            self._gaussian_params(pred, self.out[v](feat), (H, W), keep)
        return pred1, pred2

    def forward_multi(self, preds, tokens, patch_start, imgs, ref_fusion='concat'):
        """Joint N-view inference (src/multiview.py) on the tokens of one aggregator pass over all views.

        View 0 is the reference: all of its Gaussians are kept and it takes the first Gaussian head. Every other
        view takes the second head (VGGT's frames 1..S-1 share one camera token, as view 2 did in training)
        and gets a filter mask against the reference (``preds[k]['mask']``; cross-view filtering between
        the non-reference views is done in src/multiview.py). Feature aggregation into the reference:
        ``ref_fusion='concat'`` attends to the matched patches of all other views at once, ``'best'`` only
        to the view with the most matches (summed over the batch). With two views both equal ``forward``."""
        B, H, W, _ = preds[0]['pts3d'].shape
        token_mats, n_matches = [], []
        for pred in preds[1:]:
            pred['mask'], pred['cross_mat'], tm, n = self._ref_filter(preds[0], pred, (H, W))
            token_mats.append(tm)
            n_matches.append(sum(n))
        toks = [[t[:, s:s + 1] if t is not None else None for t in tokens] for s in range(len(preds))]
        if self.use_attention:
            p = patch_start
            feats = [torch.cat([tok[l][:, 0, p:] for l in TOKEN_LAYERS], dim=-1) for tok in toks]
            if ref_fusion == 'concat':
                fused = self.fusion_module.forward_multi(feats[0], feats[1:], token_mats)
            elif ref_fusion == 'best':
                k = int(np.argmax(n_matches))
                fused = self.fusion_module(feats[0], feats[1 + k], token_mats[k])
            else:
                raise ValueError(f'ref_fusion {ref_fusion!r} (concat | best)')
            fused = torch.split(fused, tokens[TOKEN_LAYERS[0]].shape[-1], dim=-1)
            toks[0] = list(toks[0])
            for l, f in zip(TOKEN_LAYERS, fused):
                toks[0][l] = torch.cat([toks[0][l][:, :, :p], f[:, None]], dim=2)

        for s, (pred, tok) in enumerate(zip(preds, toks)):
            head = 0 if s == 0 else 1
            feat = self.dpt[head](tok, imgs[:, s:s + 1], patch_start)[:, 0]                # (B, 256, 518, 518)
            self._gaussian_params(pred, self.out[head](feat), (H, W), None if s == 0 else pred['mask'])
        return preds
