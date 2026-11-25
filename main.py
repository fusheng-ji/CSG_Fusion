"""CSG-Fusion training / evaluation entry point.

One LightningModule, CSGFusion, with two backbones selected by `model.backbone` in the yaml config:

    external   frozen MASt3R ViT encoder / decoder; MASt3R point-cloud heads (frozen, or fine-tuned by the
               MASt3R confidence loss when `loss.mast3r_loss_weight` is set); CSG-Fusion Gaussian head
               (src/csg_gaussian_head.py). This is the published model.
    vggt       frozen VGGT-1B aggregator, VGGT point head (fine-tuned in the same way), the same CSG-Fusion
               head on its tokens (src/vggt_backbone.py)

The training objective (paper Sec. 4.3) is MSE + LPIPS on the rendered target views, optionally plus the same
terms on the re-rendered context views (`context_loss`) and the MASt3R confidence loss on the point maps.

Every switch falls back to DEFAULTS below, which reproduce the published model, so checkpoints written
before a switch existed rebuild unchanged. Use `load_model()` to load a checkpoint.

Usage:  python main.py configs/<experiment>.yaml [key=value ...]
"""
import json
import os
import random
import sys

import einops
import lightning as L
import lpips
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Subset


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# MASt3R / DUSt3R / PixelSplat import themselves as top-level packages
for _p in ('src/pixelsplat_src', 'src/mast3r_src', 'src/mast3r_src/dust3r'):
    _p = os.path.join(ROOT, _p)
    if _p not in sys.path:
        sys.path.append(_p)

from src.mast3r_src.dust3r.dust3r.losses import L21  # noqa: E402
from src.mast3r_src.mast3r.losses import ConfLoss, Regr3D  # noqa: E402
import data.scannetpp.scannetpp as scannetpp  # noqa: E402
import src.mast3r_src.mast3r.model as mast3r_model  # noqa: E402
import utils.compute_ssim as compute_ssim  # noqa: E402
import utils.export as export  # noqa: E402
import utils.geometry as geometry  # noqa: E402
import utils.loss_mask as loss_mask  # noqa: E402
import utils.sh_utils as sh_utils  # noqa: E402
import workspace  # noqa: E402


# --------------------------------------------------------------------------------------
# Switch defaults. A yaml only needs to set what differs; configs/base.yaml (the base of
# every experiment config) spells the switches out explicitly.
# --------------------------------------------------------------------------------------
DEFAULTS = {
    'model.backbone': 'external',          # external (MASt3R) | vggt
    'loss.mast3r_active': False,           # add mast3r_loss_weight * MASt3R confidence loss (ramped, NaN/Inf skipped)
    'loss.context_warmup': True,           # 0.3 / 0.7 / 1.0 ramp on the context loss over the first 150 steps
    'loss.resized_metrics': True,          # extra 256x256 metrics
    'eval.log_fusion_ratio': True,         # log fraction of Gaussians removed by fusion
    'opt.params': ['encoder', 'gaussian_head'],
    'data.train_subset_div': 1,            # >1: train on a random 1/div of the training set
    'data.train_subset_seed': 42,
    'trainer.checkpoint_every_epoch': True,
    'trainer.max_steps': -1,               # > 0: stop after this many optimiser steps (pilot runs)
    'trainer.resume': False,               # True: continue from <save_dir>/checkpoints/last.ckpt (needs name=<run>)
    'trainer.skip_test': False,            # True: no test-set evaluation after training (use scripts/eval/baselines)
    'loggers.use_tensorboard': True,        # TensorBoard event files in save_dir/tensorboard (local)
    'loggers.wandb_offline': True,          # with loggers.use_wandb: write the W&B run locally, never sync
    # Training fixes (README "Training fixes"): off here, as in the published model, so its checkpoint loads unchanged;
    # configs/base.yaml turns them on for new runs.
    'model.pad_detach_fix': False,          # decoder: keep view-2 gradients of items that need padding
    'model.fusion_mask_transpose_fix': False,  # FeatureFusion: view-1 queries attend to their own matched view-2 patches
    'model.fusion_skip_unmatched': False,   # FeatureFusion: unmatched queries get no (uniform) attention update
    'model.filter_union_matches': False,    # matching filter: compare with every matched view-1 patch, not the last one
    'loss.target_mask_fix': False,          # masked training MSE: each target view uses its own mask (see calculate_loss)
    # Speed / memory switches (README "Performance switches")
    'perf.elementwise_covariance': False,  # True: covariance 3x3 products element-wise instead of bmm, faster
                                           # steps; agrees to ~1 ulp, not bit for bit (changes numerics)
    'perf.checkpoint_gaussian_dpt': False, # recompute the Gaussian DPT heads in backward: less memory, same values, slower
    'perf.rope': 'torch',                  # torch: PyTorch RoPE in the MASt3R encoder / decoder | cuda: the curope kernel (needs it built; --use_fast_math, changes numerics)
    'perf.cuda_graph': True,               # eval mode, no autograd: replay the frozen MASt3R encoder blocks / decoder
                                           # from CUDA graphs (same kernels, bit-identical; extra GPU memory per
                                           # captured input shape)
    'perf.backbone_bf16': False,           # True: frozen MASt3R encoder / decoder under bf16 autocast (changes numerics)
    'perf.matching_subsample': 8,          # pixel stride of the matching-filter seeds (MASt3R's 8; 16 = 4x fewer
                                           # queries, changes which Gaussians are filtered)
    'data.pin_memory': True,               # page-locked batches: asynchronous host-to-GPU copies (same data)
    'data.prefetch_factor': None,          # batches each worker loads ahead (None: torch default 2; same data)
    'data.persistent_workers': False,      # keep workers across epochs. True changes the sampled views from
                                           # epoch 2 on (worker RNG streams continue instead of being re-seeded)
}


def cfg(config, key, default=None):
    """Read a dotted key; fall back to DEFAULTS (the published model's behaviour)."""
    if default is None:
        default = DEFAULTS.get(key)
    value = omegaconf.OmegaConf.select(config, key, default=default)
    return value


def dataloader_kwargs(config):
    """DataLoader settings that do not change which samples are drawn (except persistent_workers)."""
    kw = dict(num_workers=config.data.num_workers, pin_memory=bool(cfg(config, 'data.pin_memory')))
    if config.data.num_workers > 0:
        kw['persistent_workers'] = bool(cfg(config, 'data.persistent_workers'))
        if cfg(config, 'data.prefetch_factor') is not None:
            kw['prefetch_factor'] = int(cfg(config, 'data.prefetch_factor'))
    return kw


class CSGFusion(L.LightningModule):
    """CSG-Fusion: two images -> one compact set of 3D Gaussians (Matching-based Gaussian Filter + Mask-based
    Feature Aggregation on a frozen MASt3R or VGGT backbone), with its losses, training and test loops."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        check_switches(config)
        backbone = cfg(config, 'model.backbone')
        self.backbone = backbone
        import src.pixelsplat_src.benchmarker as benchmarker
        import src.pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder

        # ---- Backbone ------------------------------------------------------------------
        # `sh_degree` is the number of SH coefficients per colour channel, not the SH degree: the head emits
        # 3 * sh_degree SH channels and the renderer uses degree isqrt(sh_degree) - 1. The default 1 is degree 0,
        # i.e. view-independent colour.
        sh_degree = config.sh_degree if hasattr(config, 'sh_degree') else 1
        if backbone == 'external':      # frozen MASt3R encoder / decoder; its heads predict pts3d + desc
            self.encoder = mast3r_model.AsymmetricMASt3R(
                head_type='gaussian_head', output_mode='pts3d+desc24',
                pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512),
                depth_mode=('exp', -mast3r_model.inf, mast3r_model.inf),
                conf_mode=('exp', 1, mast3r_model.inf),
                enc_embed_dim=1024, enc_depth=24, enc_num_heads=16,
                dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True,
            )
            self.encoder.eval()
            self.encoder.requires_grad_(False)
        elif backbone == 'vggt':        # frozen VGGT-1B aggregator; CSG-Fusion head on its tokens (src/vggt_backbone.py)
            from src.vggt_backbone import VGGTBackbone
            self.encoder = VGGTBackbone(pretrained=cfg(config, 'model.vggt_pretrained', 'facebook/VGGT-1B'),
                                        init_log_scale=cfg(config, 'model.vggt_init_log_scale', 0.0),
                                        bf16=cfg(config, 'model.vggt_bf16', True))
            self.encoder.point_head.requires_grad_(False)
        else:
            raise ValueError(f'unknown model.backbone {backbone!r} (external | vggt)')
        self.backbone_graphs = ()
        if backbone != 'vggt':          # MASt3R backbones: RoPE kernel choice, CUDA graphs of the frozen ViT
            self._select_rope(cfg(config, 'perf.rope'))
        if backbone != 'vggt' and cfg(config, 'perf.cuda_graph'):
            import src.backbone_graph as backbone_graph
            self.backbone_graphs = backbone_graph.install(self.encoder, active=self._graphs_active)

        # ---- CSG-Fusion Gaussian head --------------------------------------------------
        if backbone == 'vggt':
            from src.vggt_backbone import VGGTGaussianHead
            self.gaussian_head = VGGTGaussianHead(use_attention=config.use_attention,
                                                  attention_neighbor=config.attention_neighbor,
                                                  use_offsets=config.use_offsets, sh_degree=sh_degree,
                                                  dist_threshold=config.dist_threshold)
            self.gaussian_head.filter_union_matches = bool(cfg(config, 'model.filter_union_matches'))
            # perf.matching_subsample / checkpoint_gaussian_dpt are for the MASt3R head only
        else:
            from src.csg_gaussian_head import GaussianHead
            self.gaussian_head = GaussianHead(
                use_attention=config.use_attention,
                attention_neighbor=config.attention_neighbor,
                use_offsets=config.use_offsets,
                sh_degree=sh_degree,
                depth_mode=('exp', -mast3r_model.inf, mast3r_model.inf),
                conf_mode=('exp', 1, mast3r_model.inf),
                dist_threshold=config.dist_threshold,
            )
            self.gaussian_head.checkpoint_dpt = bool(cfg(config, 'perf.checkpoint_gaussian_dpt'))
            self.gaussian_head.matching_subsample = int(cfg(config, 'perf.matching_subsample'))
            self.gaussian_head.filter_union_matches = bool(cfg(config, 'model.filter_union_matches'))
        if hasattr(self.gaussian_head, 'fusion_module'):
            self.gaussian_head.fusion_module.mask_transpose_fix = bool(cfg(config, 'model.fusion_mask_transpose_fix'))
            self.gaussian_head.fusion_module.skip_unmatched = bool(cfg(config, 'model.fusion_skip_unmatched'))

        # The decoder which renders the predicted Gaussians, lightly modified from PixelSplat
        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(background_color=[0.0, 0.0, 0.0])
        self.decoder.pad_detach_fix = bool(cfg(config, 'model.pad_detach_fix'))

        self.benchmarker = benchmarker.Benchmarker()

        # ---- Loss criteria -------------------------------------------------------------
        self.lpips_criterion = lpips.LPIPS(net='alex', spatial=bool(config.loss.average_over_mask))

        # A non-null mast3r_loss_weight unfreezes the point-cloud heads (MASt3R downstream_head1/2, or the VGGT
        # point head) so that the MASt3R confidence loss fine-tunes pts3d. The loss itself is only added when
        # loss.mast3r_active is also set (calculate_loss); the heads are unfrozen either way.
        if config.loss.mast3r_loss_weight is not None:
            self.mast3r_criterion = ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2)
            if backbone == 'vggt':
                self.encoder.point_head.requires_grad_(True)
            else:
                self.encoder.downstream_head1.requires_grad_(True)
                self.encoder.downstream_head2.requires_grad_(True)
            self.gaussian_head.requires_grad_(True)

        self.save_hyperparameters()

    # ==================================================================================
    # Forward
    # ==================================================================================
    def forward(self, view1, view2):
        if self.backbone == 'vggt':
            res1, res2, tokens, patch_start, imgs = self.encoder(view1, view2, view1['original_img'].shape[-2:])
            pred1, pred2 = self.gaussian_head(res1, res2, tokens, patch_start, imgs)
            pred1['covariances'] = self._covariance(pred1['scales'], pred1['rotations'])
            pred2['covariances'] = self._covariance(pred2['scales'], pred2['rotations'])
            self._add_rgb_residual(view1, view2, pred1, pred2)
            pred2['pts3d_in_other_view'] = pred2.pop('pts3d')
            pred2['means_in_other_view'] = pred2.pop('means')
            return pred1, pred2

        # Frozen MASt3R encoder + decoder
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(cfg(self.config, 'perf.backbone_bf16'))):
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

        # The point-cloud heads (pts3d + desc) run outside no_grad so that they can be fine-tuned by the MASt3R
        # loss; the CSG-Fusion head reads the same fp32 decoder tokens.
        res1 = self.encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = self.encoder._downstream_head(2, [tok.float() for tok in dec2], shape2)
        pred1, pred2 = self.gaussian_head(res1, res2, [tok.float() for tok in dec1], [tok.float() for tok in dec2],
                                          view1['original_img'], view2['original_img'])
        pred1['covariances'] = self._covariance(pred1['scales'], pred1['rotations'])
        pred2['covariances'] = self._covariance(pred2['scales'], pred2['rotations'])
        self._add_rgb_residual(view1, view2, pred1, pred2)
        pred2['pts3d_in_other_view'] = pred2.pop('pts3d')
        pred2['means_in_other_view'] = pred2.pop('means')
        return pred1, pred2

    def _graphs_active(self):
        """CUDA graphs only for inference: eval mode and not inside trainer.fit (whose validation would
        capture extra graphs at the training batch size)."""
        if self.training:
            return False
        trainer = self._trainer
        return trainer is None or str(getattr(trainer.state, 'fn', '')) not in ('fit', 'TrainerFn.FITTING')

    def _select_rope(self, kind):
        """perf.rope: 'torch' keeps the PyTorch RoPE; 'cuda' swaps in the curope kernel (same frequencies)."""
        if kind == 'torch':
            return
        if kind != 'cuda':
            raise ValueError(f'unknown perf.rope {kind!r} (torch | cuda)')
        from models.pos_embed import cuRoPE2D
        if cuRoPE2D is None:
            raise ImportError('perf.rope=cuda needs the curope extension: bash scripts/install_rasterizers.sh')
        if cfg(self.config, 'perf.backbone_bf16'):
            raise ValueError('perf.rope=cuda has no bf16 kernel; use it without perf.backbone_bf16')
        for m in self.encoder.modules():
            if getattr(m, 'rope', None) is not None:
                m.rope = cuRoPE2D(freq=m.rope.base, F0=m.rope.F0)

    def _covariance(self, scales, rotations):
        return geometry.build_covariance(scales, rotations, elementwise=cfg(self.config, 'perf.elementwise_covariance'))

    def _add_rgb_residual(self, view1, view2, pred1, pred2):
        """SH = predicted residual + RGB of the input pixel.

        The input pixel's colour is converted to SH and added to the DC coefficient (sh[..., 0]), so the head
        predicts a colour residual: with a zero residual each Gaussian takes the colour of the pixel it was
        predicted from (degree-0 SH evaluates to C0 * sh + 0.5 = rgb)."""
        learn_residual = not hasattr(self.config, 'learn_residual') or self.config.learn_residual
        if learn_residual:
            new_sh1 = torch.zeros_like(pred1['sh'])
            new_sh2 = torch.zeros_like(pred2['sh'])
            new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1['original_img'], 'b c h w -> b h w c'))
            new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2['original_img'], 'b c h w -> b h w c'))
            pred1['sh'] = pred1['sh'] + new_sh1
            pred2['sh'] = pred2['sh'] + new_sh2

    # ==================================================================================
    # Steps
    # ==================================================================================
    def _render(self, batch, pred1, pred2, need_context=False):
        """Returns the target renders and the context-view renders (or None).

        need_context: also render the context views (the training / validation loss asks for them when
        `context_loss` is on; the test step and the baseline harness only use the target renders)."""
        need_context = need_context and bool(getattr(self.config, 'context_loss', False))
        out = self.decoder(batch, pred1, pred2, render_context=need_context)
        return out[0], (out[1] if len(out) > 1 else None)

    def _train_val_step(self, prefix, batch):
        view1, view2 = batch['context']
        pred1, pred2 = self.forward(view1, view2)
        color, context_color = self._render(batch, pred1, pred2, need_context=True)
        if prefix == 'val':
            torch.cuda.empty_cache()
        mask = loss_mask.calculate_loss_mask(batch)
        out = self.calculate_loss(
            batch, view1, view2, pred1, pred2, color, context_color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=False,
        )
        self.log_metrics(prefix, out)
        if prefix == 'val':
            torch.cuda.empty_cache()
        return out['loss']

    def training_step(self, batch, batch_idx):
        return self._train_val_step('train', batch)

    def validation_step(self, batch, batch_idx):
        return self._train_val_step('val', batch)

    def test_step(self, batch, batch_idx):
        view1, view2 = batch['context']
        num_targets = len(batch['target'])

        with self.benchmarker.time("encoder"):
            pred1, pred2 = self.forward(view1, view2)
        with self.benchmarker.time("decoder", num_calls=num_targets):
            color, _ = self._render(batch, pred1, pred2)

        fusion_ratio = None
        if cfg(self.config, 'eval.log_fusion_ratio') and 'mask' in pred2:
            fusion_ratio = torch.sum(pred2['mask'] == 0).item() / pred2['mask'].numel()

        if 'dtu' in batch['context'][0].get('dataset', ''):
            mask = None  # DTU: evaluate against the full, un-masked image
        elif self.config.loss.apply_mask:
            mask = loss_mask.calculate_loss_mask(batch)
        else:
            mask = None
        if mask is not None and self.config.loss.apply_mask and mask.sum() == 0:
            # no ground-truth pixel is visible from the context views: nothing to score (the baseline
            # evaluation in scripts/eval/baselines skips the same samples)
            print(f"Skipping test sample with an empty mask: {batch['scene'][0]}")
            return None

        out = self.calculate_loss(
            batch, view1, view2, pred1, pred2, color, None, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=True,
        )
        if fusion_ratio is not None:
            out['fusion_ratio'] = fusion_ratio
        self.log_metrics('test', out)
        return out['loss'], -10 * out['mse'].log10(), out['lpips'], out['ssim']

    def on_test_start(self):
        self.benchmarker = type(self.benchmarker)()       # one benchmark.json per test run

    def on_test_end(self):
        if self.trainer.is_global_zero:
            self.benchmarker.dump(os.path.join(self.config.save_dir, "benchmark.json"))

    def teardown(self, stage):
        # the Trainer moves the model to the CPU and frees the cuBLAS workspaces next: captured CUDA graphs would
        # replay freed memory in the next trainer.test
        for graphed in self.backbone_graphs:
            graphed.clear()

    # ==================================================================================
    # Loss
    # ==================================================================================
    def calculate_loss(self, batch, view1, view2, pred1, pred2, color, context_color, mask,
                       apply_mask=True, average_over_mask=True, calculate_ssim=False):
        """Returns a dict: loss, mse, lpips [, ssim] [, *_resized]."""
        out = {}
        resized = cfg(self.config, 'loss.resized_metrics')

        target_color = torch.stack([target_view['original_img'] for target_view in batch['target']], dim=1)
        predicted_color = color.clamp(0, 1) if calculate_ssim else color    # test metrics: displayable colours

        if apply_mask:
            assert mask.sum() > 0, "There are no valid pixels in the mask!"
            target_color = target_color * mask[..., None, :, :]
            predicted_color = predicted_color * mask[..., None, :, :]

        flattened_color = einops.rearrange(predicted_color, 'b v c h w -> (b v) c h w')
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        if mask is not None:
            flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w')
        if resized:
            # logged only, not part of the loss: detached, so the resized metrics record no autograd graph
            flattened_color_resized = F.interpolate(flattened_color.detach(), size=(256, 256), mode='bilinear', align_corners=False)
            flattened_target_color_resized = F.interpolate(flattened_target_color, size=(256, 256), mode='bilinear', align_corners=False)
            if average_over_mask:
                # (bv, h, w) -> (bv, 1, h, w) for bilinear interpolation, then back
                flattened_mask_resized = F.interpolate(flattened_mask[:, None].float(), size=(256, 256), mode='bilinear',
                                                       align_corners=False)[:, 0]

        loss = 0

        # Loss ramp: weight 0.3 for global steps < 50, 0.7 for < 150, then 1.0. It always scales the MASt3R
        # loss; it scales the context-view loss only with loss.context_warmup. The target-view loss is not ramped.
        current_step = self.trainer.global_step
        ramp = 0.3 if current_step < 50 else 0.7 if current_step < 150 else 1.0
        weight_scale = ramp if cfg(self.config, 'loss.context_warmup') else 1.0

        # MSE
        rgb_l2_loss = (predicted_color - target_color) ** 2
        if average_over_mask:
            # (b, v, c, h, w) x mask (b, v, h, w). mask[:, None] (as in Splatt3R) multiplies colour channel j of every
            # view by the mask of view j; loss.target_mask_fix gives every view its own mask. Same for one target view.
            view_mask = mask[:, :, None] if cfg(self.config, 'loss.target_mask_fix') else mask[:, None]
            mse_loss = (rgb_l2_loss * view_mask).sum() / mask.sum()
        else:
            mse_loss = rgb_l2_loss.mean()
        if resized:
            rgb_l2_loss_resized = (flattened_color_resized - flattened_target_color_resized) ** 2
            if average_over_mask:
                mse_loss_resized = (rgb_l2_loss_resized * flattened_mask_resized[:, None, ...]).sum() / flattened_mask_resized.sum()
            else:
                mse_loss_resized = rgb_l2_loss_resized.mean()

        # LPIPS
        lpips_loss = self.lpips_criterion(flattened_target_color, flattened_color, normalize=True)
        if average_over_mask:
            lpips_loss = (lpips_loss * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
        else:
            lpips_loss = lpips_loss.mean()
        if resized:
            with torch.no_grad():
                lpips_loss_resized = self.lpips_criterion(flattened_target_color_resized, flattened_color_resized, normalize=True)
            if average_over_mask:
                lpips_loss_resized = (lpips_loss_resized * flattened_mask_resized[:, None, ...]).sum() / flattened_mask_resized.sum()
            else:
                lpips_loss_resized = lpips_loss_resized.mean()

        # Context-view reconstruction loss
        if hasattr(self.config, 'context_loss') and self.config.context_loss and context_color is not None:
            gt_context_color = torch.stack([context_view['original_img'] for context_view in batch['context']], dim=1)
            flattened_pred_context_color = einops.rearrange(context_color, 'b v c h w -> (b v) c h w')
            flattened_gt_context_color = einops.rearrange(gt_context_color, 'b v c h w -> (b v) c h w')
            context_mse_loss = ((context_color - gt_context_color) ** 2).mean()
            context_lpips_loss = self.lpips_criterion(flattened_gt_context_color, flattened_pred_context_color, normalize=True).mean()
            # Added into the mse / lpips terms, so the logged train / val mse, psnr and lpips mix target and
            # context views whenever context_loss is on (the test step renders no context views).
            mse_loss += context_mse_loss * weight_scale
            lpips_loss += context_lpips_loss * weight_scale

        loss += self.config.loss.mse_loss_weight * mse_loss
        loss += self.config.loss.lpips_loss_weight * lpips_loss
        if resized:
            loss_resized = 0
            loss_resized += self.config.loss.mse_loss_weight * mse_loss_resized
            loss_resized += self.config.loss.lpips_loss_weight * lpips_loss_resized

        # MASt3R loss
        # (DTU views have no ground-truth points)
        if cfg(self.config, 'loss.mast3r_active') and self.config.loss.mast3r_loss_weight is not None and 'pts3d' in view1:
            # fine-tune the point-cloud heads with a ramped MASt3R confidence loss; a NaN / Inf value is
            # skipped for that step (not clipped), so it contributes neither loss nor gradient
            mast3r_loss = self.mast3r_criterion(view1, view2, pred1, pred2)[0]
            if not torch.isnan(mast3r_loss) and not torch.isinf(mast3r_loss):
                loss += self.config.loss.mast3r_loss_weight * mast3r_loss * ramp

        out.update(loss=loss, mse=mse_loss, lpips=lpips_loss)
        if resized:
            out.update(loss_resized=loss_resized, mse_resized=mse_loss_resized, lpips_resized=lpips_loss_resized)

        # Masked SSIM
        if calculate_ssim:
            if average_over_mask:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=True)
                ssim_val = (ssim_val * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
            else:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=False)
                ssim_val = ssim_val.mean()
            out['ssim'] = ssim_val
            if resized:
                if average_over_mask:
                    ssim_val_resized = compute_ssim.compute_ssim(flattened_target_color_resized, flattened_color_resized, full=True)
                    ssim_val_resized = (ssim_val_resized * flattened_mask_resized[:, None, ...]).sum() / flattened_mask_resized.sum()
                else:
                    ssim_val_resized = compute_ssim.compute_ssim(flattened_target_color_resized, flattened_color_resized, full=False)
                    ssim_val_resized = ssim_val_resized.mean()
                out['ssim_resized'] = ssim_val_resized
            if average_over_mask:
                # the masked sums run over three colour channels but are divided by the number of pixels: report means
                for key in ('mse', 'mse_resized', 'ssim', 'ssim_resized'):
                    if key in out:
                        out[key] = out[key] / target_color.shape[2]
        out['batch_size'] = target_color.shape[0]
        return out

    def log_metrics(self, prefix, out):
        values = {
            f'{prefix}/loss': out['loss'],
            f'{prefix}/mse': out['mse'],
            f'{prefix}/psnr': -10.0 * out['mse'].log10(),
            f'{prefix}/lpips': out['lpips'],
        }
        if 'mse_resized' in out:
            values[f'{prefix}/loss_resized'] = out['loss_resized']
            values[f'{prefix}/mse_resized'] = out['mse_resized']
            values[f'{prefix}/lpips_resized'] = out['lpips_resized']
            values[f'{prefix}/psnr_resized'] = -10.0 * out['mse_resized'].log10()
        if 'ssim' in out:
            values[f'{prefix}/ssim'] = out['ssim']
        if 'ssim_resized' in out:
            values[f'{prefix}/ssim_resized'] = out['ssim_resized']
        if 'fusion_ratio' in out:
            values[f'{prefix}/fusion_ratio'] = out['fusion_ratio']
        prog_bar = prefix != 'val'
        sync_dist = prefix != 'train'
        self.log_dict(values, prog_bar=prog_bar, sync_dist=sync_dist, batch_size=out['batch_size'])

    # ==================================================================================
    # Optimizer
    # ==================================================================================
    def configure_optimizers(self):
        # Adam over every parameter of the submodules named in opt.params (default: encoder, gaussian_head).
        # Only those with requires_grad receive gradients and change: the frozen ViT encoder / decoder (or VGGT
        # aggregator) stays fixed, the point-cloud heads train only when mast3r_loss_weight unfroze them, and the
        # VGGT backbone's global log_scale has gradients only through the MASt3R loss.
        params = []
        for name in cfg(self.config, 'opt.params'):
            params += list(getattr(self, name).parameters())
        optimizer = torch.optim.Adam(params, lr=self.config.opt.lr)

        def lr_lambda(step):  # per-step warm-up, then x0.1 at 2k / 20k / 40k steps
            warmup_steps = 20
            if step < warmup_steps:
                return 0.1 + 0.9 * (step / warmup_steps)
            elif step < 2000:
                return 1.0
            elif step < 20000:
                return 0.1
            elif step < 40000:
                return 0.01
            else:
                return 0.001
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def _strategy(devices):
    """DDP only when more than one GPU is used. `devices` is a Lightning devices value: an int, a list of ids,
    'auto' or -1 (every visible GPU), or a string of ids such as '0,1'."""
    if isinstance(devices, str):
        n = torch.cuda.device_count() if devices == 'auto' else len([d for d in devices.split(',') if d.strip()])
    elif isinstance(devices, int):
        n = torch.cuda.device_count() if devices == -1 else devices
    else:
        n = len(devices)
    return 'ddp_find_unused_parameters_true' if n > 1 else 'auto'


# ======================================================================================
# Checkpoint loading
# ======================================================================================
# Config keys of variants that are not in this repository, with the only value that still loads (the default).
REMOVED_SWITCHES = {'model.fusion_symmetric': 'none', 'model.soft_fusion': False, 'model.dist_threshold_rel': None,
                    'model.fusion_count_bias': False, 'loss.context_mask': False, 'loss.lpips_net': 'alex',
                    'with_matching_based_rgb_residual': False}


def check_switches(config):
    for key, default in REMOVED_SWITCHES.items():
        value = omegaconf.OmegaConf.select(config, key, default=default)
        if value != default:
            raise ValueError(f'{key}={value!r} is not supported; only {default!r} loads')


def load_model(ckpt_path, device='cuda', overrides=None, strict=True):
    """Load a CSG-Fusion checkpoint (MASt3R or VGGT backbone). `overrides`: {dotted key: value}."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['hyper_parameters']['config']
    if not isinstance(config, omegaconf.DictConfig):
        config = omegaconf.OmegaConf.create(config)
    if overrides:
        omegaconf.OmegaConf.set_struct(config, False)
        for key, value in overrides.items():
            omegaconf.OmegaConf.update(config, key, value, merge=False)
    if cfg(config, 'model.backbone') == 'vggt':     # the checkpoint holds every weight: skip the VGGT-1B download
        omegaconf.OmegaConf.update(config, 'model.vggt_pretrained', '', force_add=True)
    model = CSGFusion(config)
    model.load_state_dict(ckpt['state_dict'], strict=strict)
    return model.to(device)


# ======================================================================================
# Training
# ======================================================================================
def run_experiment(config):

    L.seed_everything(config.seed, workers=True)

    # Loggers
    os.makedirs(os.path.join(config.save_dir, config.name), exist_ok=True)
    loggers = []
    if cfg(config, 'loggers.use_tensorboard'):  # `tensorboard --logdir <save_dir>/tensorboard`
        loggers.append(L.pytorch.loggers.TensorBoardLogger(save_dir=os.path.join(config.save_dir, 'tensorboard'),
                                                           name=config.name, default_hp_metric=False))
    if config.loggers.use_csv_logger:
        loggers.append(L.pytorch.loggers.CSVLogger(save_dir=config.save_dir, name=config.name))
    if config.loggers.use_wandb:
        offline = bool(cfg(config, 'loggers.wandb_offline'))
        wandb_logger = L.pytorch.loggers.WandbLogger(
            project=cfg(config, 'loggers.wandb_project', 'csg_fusion'),
            name=config.name,
            save_dir=config.save_dir,
            config=omegaconf.OmegaConf.to_container(config),
            offline=offline,  # offline: the run stays in save_dir/wandb (sync later with `wandb sync` if wanted)
        )
        loggers.append(wandb_logger)

    # Profiler
    if config.use_profiler:
        profiler = L.pytorch.profilers.PyTorchProfiler(
            dirpath=config.save_dir,
            filename='trace',
            export_to_chrome=True,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.save_dir),
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
            with_stack=True,
        )
    else:
        profiler = None

    # Model
    print('Loading Model')
    model = CSGFusion(config)
    if config.use_pretrained:
        # MASt3R's release pickles an argparse Namespace, so it cannot load with PyTorch >= 2.6's weights_only default
        ckpt = torch.load(config.pretrained_mast3r_path, map_location='cpu', weights_only=False)
        _ = model.encoder.load_state_dict(ckpt['model'], strict=False)
        del ckpt

    # Datasets
    print('Building Datasets')
    train_dataset = scannetpp.get_scannet_dataset(
        config.data.root,
        'train',
        config.data.resolution,
        num_epochs_per_epoch=config.data.epochs_per_train_epoch,
    )
    subset_div = cfg(config, 'data.train_subset_div')
    if subset_div > 1:
        seed = cfg(config, 'data.train_subset_seed')
        random.seed(seed)
        np.random.seed(seed)
        dataset_size = len(train_dataset)
        indices = random.sample(range(dataset_size), dataset_size // subset_div)
        train_dataset = Subset(train_dataset, indices)
        print(f"Training on 1/{subset_div} of the data: {len(train_dataset)} / {dataset_size}")

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.data.batch_size,
        **dataloader_kwargs(config),
    )
    val_dataset = scannetpp.get_scannet_test_dataset(
        config.data.root,
        alpha=0.5,
        beta=0.5,
        resolution=config.data.resolution,
        use_every_n_sample=100,
    )
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config.data.batch_size,
        **dataloader_kwargs(config),
    )

    # Training
    print('Training')
    callbacks = [
        L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
        export.SaveBatchData(save_dir=config.save_dir),
    ]
    if cfg(config, 'trainer.checkpoint_every_epoch'):
        callbacks.append(ModelCheckpoint(
            dirpath=os.path.join(config.save_dir, 'checkpoints'),
            filename='csg-fusion-{epoch}',
            save_top_k=-1,
            every_n_epochs=1,
            save_last=True,
        ))
    trainer = L.Trainer(
        accelerator="gpu",
        benchmark=True,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        default_root_dir=config.save_dir,
        devices=config.devices,
        gradient_clip_val=config.opt.gradient_clip_val,
        log_every_n_steps=1,
        logger=loggers,
        max_epochs=config.opt.epochs,
        max_steps=cfg(config, 'trainer.max_steps'),
        profiler=profiler,
        strategy=_strategy(config.devices),
    )
    resume_from = os.path.join(config.save_dir, 'checkpoints', 'last.ckpt')
    if cfg(config, 'trainer.resume'):
        if not os.path.isfile(resume_from):
            raise FileNotFoundError(f'trainer.resume: no {resume_from}. Set name=<name of the run to continue>')
        print(f'Resuming from {resume_from}')
    else:
        resume_from = None
    trainer.fit(model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val, ckpt_path=resume_from)
    if cfg(config, 'trainer.skip_test'):
        return

    # Testing
    original_save_dir = config.save_dir
    results = {}
    for alpha, beta in ((0.9, 0.9), (0.7, 0.7), (0.5, 0.5), (0.3, 0.3)):

        test_dataset = scannetpp.get_scannet_test_dataset(
            config.data.root,
            alpha=alpha,
            beta=beta,
            resolution=config.data.resolution,
            use_every_n_sample=10,
        )
        data_loader_test = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=1,                # PSNR per image: a larger batch would pool the squared error over images
            **dataloader_kwargs(config),
        )

        # Both masking modes are reported (apply_mask, average_over_mask): (True, False) zeroes the pixels not
        # visible from the context views and averages over the full image (the metric of the
        # README tables); (True, True) averages over the visible pixels only.
        masking_configs = ((True, False), (True, True))
        for apply_mask, average_over_mask in masking_configs:

            new_save_dir = os.path.join(
                original_save_dir,
                f'alpha_{alpha}_beta_{beta}_apply_mask_{apply_mask}_average_over_mask_{average_over_mask}'
            )
            os.makedirs(new_save_dir, exist_ok=True)
            model.config.save_dir = new_save_dir

            L.seed_everything(config.seed, workers=True)

            trainer = L.Trainer(
                accelerator="gpu",
                benchmark=True,
                callbacks=[export.SaveBatchData(save_dir=new_save_dir)],
                default_root_dir=new_save_dir,
                devices=config.devices,
                log_every_n_steps=1,
                strategy=_strategy(config.devices),
            )

            model.lpips_criterion = lpips.LPIPS(net='alex', spatial=average_over_mask)
            model.config.loss.apply_mask = apply_mask
            model.config.loss.average_over_mask = average_over_mask
            res = trainer.test(model, dataloaders=data_loader_test)
            results[f"alpha: {alpha}, beta: {beta}, apply_mask: {apply_mask}, average_over_mask: {average_over_mask}"] = res

            if trainer.is_global_zero:
                with open(os.path.join(original_save_dir, 'results.json'), 'w') as f:
                    json.dump(results, f)


if __name__ == "__main__":
    config_path = os.path.abspath(sys.argv[1])
    os.chdir(ROOT)  # configs use paths relative to the repo root (./checkpoint, ./data, results/)

    # Load the config, create a results directory at config.save_dir, etc.
    config = workspace.load_config(config_path, sys.argv[2:])
    if os.getenv("LOCAL_RANK", '0') == '0':
        config = workspace.create_workspace(config)
        os.environ['CSG_FUSION_RUN_NAME'] = config.name
    else:  # the other ranks of a multi-GPU run are started by rank 0 and use its run name
        config.name = os.environ.get('CSG_FUSION_RUN_NAME', config.name)

    run_experiment(config)
