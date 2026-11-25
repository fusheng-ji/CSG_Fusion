"""CSG-Fusion Gaussian head on the MASt3R backbone (``model.backbone: external``, the published model).

Given the MASt3R predictions of the two context views (``pts3d``, ``desc``) and their decoder tokens, the head
predicts one pixel-aligned 3D Gaussian per pixel and view, and combines the paper's two components:

* Matching-based Gaussian Filter (Sec. 4.1): reciprocal descriptor matches between the views, grouped into a
  16 x 16-patch matching matrix; view-2 pixels whose 3D point lies within ``dist_threshold`` of the matched
  view-1 patch are marked redundant (``pred2['mask'] == 0``) and their Gaussians are dropped
  (``src/fast_matching.py``).
* Mask-based Feature Aggregation (Sec. 4.2): view-1 decoder tokens attend to the view-2 tokens of their matched
  patches (``src/feature_aggregation.FeatureFusion``) before the Gaussian DPT heads.

Two DPT heads (``src/dpt_gs_head.py``), one per view, then predict per pixel a mean offset, scale, rotation,
SH colour and opacity.
"""
from src.dpt_gs_head import create_gs_dpt_head, run_dpt
import src.fast_matching as fast_matching
import torch
import torch.nn as nn
from einops import rearrange
from src.feature_aggregation import FeatureFusion


class GaussianHead(nn.Module):
    """Matching-based Gaussian Filter + Mask-based Feature Aggregation + two Gaussian DPT heads.

    forward(pred1, pred2, decout1, decout2, img1, img2) adds to the MASt3R prediction dicts ``scales``,
    ``rotations``, ``sh`` (B, H, W, 3, sh_degree), ``opacities`` and ``means``, and to ``pred2`` the filter
    output ``mask`` (B, H, W, 1; 1 = keep, 0 = redundant) and ``cross_mat`` (B, N, N), the patch match counts.

    Args:
        use_attention: build and apply the Mask-based Feature Aggregation module (``fusion_module``).
        attention_neighbor: neighbourhood radius (in patches) by which the matching matrix is dilated.
        use_offsets: means = pts3d + a predicted offset (False: means = pts3d).
        sh_degree: number of SH coefficients per colour channel (the head predicts 3 * sh_degree SH channels;
            1 = degree-0, view-independent colour).
        dist_threshold: filter distance in the unit of the MASt3R points (metres).
        depth_mode, conf_mode: forwarded to the DPT heads; unused, since ``run_dpt`` skips their postprocess.
    """

    def __init__(self,
                 use_attention=False,
                 attention_neighbor=0,
                 use_offsets=True,
                 sh_degree=1,
                 dist_threshold=0.1,
                 depth_mode=None, conf_mode=None,
                 ):
        super().__init__()
        self.checkpoint_dpt = False  # set from perf.checkpoint_gaussian_dpt (memory switch)
        self.matching_subsample = 8      # set from perf.matching_subsample (MASt3R's seed stride)
        self.filter_union_matches = False  # set from model.filter_union_matches
        # args
        self.use_attention = use_attention
        self.attention_neighbor = attention_neighbor
        self.use_offsets = use_offsets
        self.dist_threshold = dist_threshold
        # Gaussian channels per pixel = mean offset (3) + scales (3) + rotation quaternion (4)
        #                               + SH (3 * sh_degree) + opacity (1)
        self.d_means = 3
        self.d_scales = 3
        self.d_rotations = 4
        self.sh_degree = sh_degree
        self.d_opacities = 1
        self.gaussian_num_channels = (self.d_means + self.d_scales + self.d_rotations + self.d_opacities + 3 * self.sh_degree)

        self.gaussian_dpt_1 = create_gs_dpt_head(has_conf=False,
                                                 out_nchan=self.gaussian_num_channels,
                                                 depth_mode=depth_mode,
                                                 conf_mode=conf_mode)
        self.gaussian_dpt_2 = create_gs_dpt_head(has_conf=False,
                                                 out_nchan=self.gaussian_num_channels,
                                                 depth_mode=depth_mode,
                                                 conf_mode=conf_mode)

        # Initialisation of the last conv layer of both heads, per parameter group (same values as Splatt3R):
        # (channels, xavier gain, bias). Near-zero weights for offsets and scales give mean offsets ~0 and
        # scales ~exp(-7) ~= 0.9 mm (MASt3R points are metric); rotations and SH get unit-gain random weights;
        # the opacity bias -2 centres the initial opacities around sigmoid(-2) ~= 0.12.
        final_conv_layer_1 = self.gaussian_dpt_1.dpt.head[-1]
        final_conv_layer_2 = self.gaussian_dpt_2.dpt.head[-1]
        splits_and_inits = [
            (3, 0.001, 0.001),  # 3D mean offsets
            (3, 0.00003, -7.0),  # Scales
            (4, 1.0, 0.0),  # Rotations
            (3 * self.sh_degree, 1.0, 0.0),  # Spherical Harmonics
            (1, 1.0, -2.0)  # Opacity
        ]
        start_channels = 0
        for out_channel, s, b in splits_and_inits:
            torch.nn.init.xavier_uniform_(
                final_conv_layer_1.weight[start_channels:start_channels+out_channel, :, :, :], s
            )
            torch.nn.init.constant_(
                final_conv_layer_1.bias[start_channels:start_channels+out_channel], b
            )
            torch.nn.init.xavier_uniform_(
                final_conv_layer_2.weight[start_channels:start_channels+out_channel, :, :, :], s
            )
            torch.nn.init.constant_(
                final_conv_layer_2.bias[start_channels:start_channels+out_channel], b
            )
            start_channels += out_channel

        self.gaussian_dpt_1.dpt.requires_grad_(True)
        self.gaussian_dpt_2.dpt.requires_grad_(True)

        # Feature aggregation input: per patch token, the concatenation of the four DPT hook layers
        # (decout[0]: encoder output, 1024 channels; decout[6], decout[9], decout[12]: decoder layers, 768 each),
        # 1024 + 3 * 768 = 3328 channels, over the 32 x 32 = 1024 patch tokens of a 512 x 512 image.
        if self.use_attention:
            self.fusion_module = FeatureFusion(
                        input_dim=1024+768+768+768,         # concatenated hook channels (3328)
                        hidden_dim=512,         # hidden dim
                        output_dim=1024+768+768+768,        # output dim, same as input
                        num_heads=8,            # attention heads
                        dropout=0.1,            # dropout rate
                        neighborhood_size=self.attention_neighbor,    # neighborhood size
                        grid_size=32,           # grid size (default 32x32=1024 patches)
                    )
            self.fusion_module.requires_grad_(True)

        # Activation functions
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid

    def reg_offsets(self, xyz, shift=6.0):
        """
        Apply an activation function to the offsets so that they are small at initialization:
        direction of ``xyz`` times exp(|xyz| - shift) - exp(-shift), i.e. about exp(-6) |xyz| for small ``xyz``
        (0 at xyz = 0), growing exponentially for large ones. Same function as Splatt3R's reg_dense_offsets.
        """
        d = xyz.norm(dim=-1, keepdim=True)
        xyz = xyz / d.clip(min=1e-8)
        offsets = xyz * (torch.exp(d - shift) - torch.exp(torch.zeros_like(d) - shift))
        return offsets

    def forward(self, pred1, pred2, decout1, decout2, img1, img2):

        B, _, _, _ = pred1["pts3d"].shape
        # ---- Matching-based Gaussian Filter (Sec. 4.1) --------------------------------------------------
        desc1, desc2 = pred1['desc'].detach(), pred2['desc'].detach()
        counts = torch.stack([fast_matching.patch_counts(*fast_matching.reciprocal_matches(
            desc1[i], desc2[i], subsample=self.matching_subsample)) for i in range(B)])
        # distance_mask is True = redundant; invert so that 1 = keep
        masks = [~fast_matching.distance_mask(pred1["pts3d"][i].detach(), pred2["pts3d"][i].detach(), counts[i],
                                              threshold=self.dist_threshold, union=self.filter_union_matches) for i in range(B)]
        masks_cuda = torch.stack(masks, dim=0).float().unsqueeze(3).detach()

        pred2['mask'] = masks_cuda
        pred2['cross_mat'] = counts

        # ---- Mask-based Feature Aggregation (Sec. 4.2) --------------------------------------------------
        # Only the view-1 tokens are updated: view 1 is the reference frame whose Gaussians are all
        # kept, view 2 is the one being filtered. The fused tokens replace the four DPT hook layers.
        if self.use_attention:
            v1_features = torch.cat([decout1[hook] for hook in [0, 6, 9, 12]], dim=-1)
            v2_features = torch.cat([decout2[hook] for hook in [0, 6, 9, 12]], dim=-1)
            fused_v1_features = self.fusion_module(v1_features, v2_features, counts)
            decout1[0], decout1[6], decout1[9], decout1[12] = torch.split(fused_v1_features, [1024, 768, 768, 768], dim=-1)

        # ---- Gaussian parameters -------------------------------------------------------------------------
        gaussian_features_1 = run_dpt(self.gaussian_dpt_1.dpt, decout1, img1, checkpoint=self.checkpoint_dpt)
        gaussian_features_2 = run_dpt(self.gaussian_dpt_2.dpt, decout2, img2, checkpoint=self.checkpoint_dpt)

        offset_1, scales_1, rotations_1, sh_1, opacities_1 = torch.split(gaussian_features_1.permute(0, 2, 3, 1) , [3, 3, 4, 3 * self.sh_degree, 1], dim=-1)
        offset_2, scales_2, rotations_2, sh_2, opacities_2 = torch.split(gaussian_features_2.permute(0, 2, 3, 1) , [3, 3, 4, 3 * self.sh_degree, 1], dim=-1)

        offset_1 = self.reg_offsets(offset_1)
        offset_2 = self.reg_offsets(offset_2)

        pred1['scales'] = self.scale_activation(scales_1)
        pred2['scales'] = self.scale_activation(scales_2)

        pred1['rotations'] = rotations_1 / (rotations_1.norm(dim=-1, keepdim=True) + 1e-8)
        pred2['rotations'] = rotations_2 / (rotations_2.norm(dim=-1, keepdim=True) + 1e-8)

        pred1['sh'] = rearrange(sh_1, '... (xyz d_sh) -> ... xyz d_sh', xyz=3)
        pred2['sh'] = rearrange(sh_2, '... (xyz d_sh) -> ... xyz d_sh', xyz=3)

        pred1['opacities'] = self.opacity_activation(opacities_1)
        pred2['opacities'] = self.opacity_activation(opacities_2)
        # Filtered (mask == 0) view-2 Gaussians get opacity 0 and no gradient. The decoder
        # (src/pixelsplat_src/decoder_splatting_cuda.py) also drops them from the rendered set, so this
        # only matters for consumers of pred2 that do not apply the mask. torch.where gives the
        # same values and gradients as the boolean index_put it replaces, without a host sync.
        pred2['opacities'] = torch.where(masks_cuda == 0, 0.0, pred2['opacities'])

        # Means start at the MASt3R points, detached: the rendering losses never update the point heads
        # (only the MASt3R confidence loss does, see main.py); the predicted offsets let rendering refine
        # the Gaussian positions.
        if self.use_offsets:
            pred1['means'] = pred1["pts3d"].detach() + offset_1
            pred2['means'] = pred2["pts3d"].detach() + offset_2
        else:
            pred1['means'] = pred1["pts3d"].detach()
            pred2['means'] = pred2["pts3d"].detach()

        return pred1, pred2
