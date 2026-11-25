"""Gaussian-parameter DPT heads of the CSG-Fusion head (MASt3R backbone).

A DPT head (DUSt3R / CroCo ``DPTOutputAdapter``) maps the encoder output and three decoder layers of one
view to a full-resolution map of raw per-pixel Gaussian parameters (mean offset, scale, rotation, SH,
opacity); the activations are applied in ``src/csg_gaussian_head.py``. The adapter adds a 7x7 convolution of
the input RGB image to the last, full-resolution feature map (``input_merger``).
"""
from einops import rearrange
from typing import List
import torch
import torch.nn as nn
import torch.utils.checkpoint
from src.mast3r_src.dust3r.croco.models.dpt_block import DPTOutputAdapter, Interpolate
from src.mast3r_src.dust3r.dust3r.heads.postprocess import postprocess


class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weights, and fix forward for dust3r.

    In addition, ``input_merger`` (7x7 conv + ReLU on the RGB image, 256 channels) is added to ``path_1`` after
    it is upsampled 2x to the image resolution, which injects full-resolution image detail that the
    16-px-patch tokens do not carry, before the output head.
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

        self.feat_up = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        self.input_merger = nn.Sequential(
            nn.Conv2d(3, 256, 7, 1, 3),
            nn.ReLU(),
        )

    def forward(self, encoder_tokens: List[torch.Tensor], depths, imgs, image_size=None, conf=None):
        """encoder_tokens: per-layer token list (indexed by ``self.hooks``); imgs: (B, 3, H, W) input image.
        ``depths`` and ``conf`` are accepted for interface compatibility and unused."""
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # Full-resolution image features added to the upsampled DPT features
        direct_img_feat = self.input_merger(imgs)
        path_1 = self.feat_up(path_1)
        path_1 = path_1 + direct_img_feat

        # Output head
        out = self.head(path_1)

        return out


class PixelwiseTaskWithDPT(nn.Module):
    """DPT module for dust3r; here it returns the raw Gaussian-parameter maps (B, num_channels, H, W) for all
    pixels. The CSG-Fusion head calls ``.dpt`` directly through ``run_dpt``, so ``postprocess`` /
    ``depth_mode`` / ``conf_mode`` are never applied; the activations live in the Gaussian head."""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, depth_mode=None, conf_mode=None, **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, depths=None, imgs=None, image_size=(512, 512), conf=None):
        out = self.dpt(x, depths, imgs, image_size=(image_size[0], image_size[1]), conf=conf)
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out


def create_gs_dpt_head(has_conf=False, out_nchan=3, postprocess_func=postprocess, depth_mode=(), conf_mode=()):
    """
    return PixelwiseTaskWithDPT for given net params

    Hooks [0, 6, 9, 12] with token dims [1024, 768, 768, 768]: the MASt3R ViT-L encoder output and decoder
    layers 6, 9 and 12 (the same hooks as DUSt3R's create_dpt_head for a 12-layer decoder).
    depth_mode / conf_mode are only stored (``run_dpt`` never applies the postprocess).
    """
    l2 = 12
    feature_dim = 256
    last_dim = feature_dim//2
    ed = 1024
    dd = 768
    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess_func,
                                depth_mode=depth_mode,
                                conf_mode=conf_mode,
                                head_type='gs_params')


def run_dpt(dpt, decout, imgs, checkpoint=False, image_size=(512, 512)):
    """dpt(decout, depths=None, imgs=imgs, image_size=image_size), optionally with activation checkpointing.

    checkpoint=True keeps only the inputs and recomputes the head's full-resolution activations in
    backward: the same values and gradients for one extra head forward (the gs_params head's Dropout(0.1)
    draws the same mask again, because torch.utils.checkpoint restores the RNG state for the recomputation)."""
    if checkpoint and torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(dpt, decout, None, imgs, image_size, use_reentrant=False)
    return dpt(decout, depths=None, imgs=imgs, image_size=image_size)
