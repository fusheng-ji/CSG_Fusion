"""DepthSplat (Xu et al., CVPR 2025), https://github.com/cvg/depthsplat

DepthSplat is built on MVSplat's code base, so it runs through ``Re10kFamilyEvaluator`` unchanged apart from
the model size: the released large RealEstate10K model (``depthsplat-gs-large-re10k-256x256-view2``) is built
with the overrides of the upstream README's "evaluate the large model" command (Table 1 of the paper). Its
``+experiment=re10k`` config gives 256x256 images, near / far = 0.5 / 100 and no baseline-1 rescaling, i.e. the
GT context and target poses are used in their metric ScanNet++ scale, exactly as ``DatasetRE10k`` passes the
RE10K poses. The monocular branch is built with ``torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')``
(its ImageNet weights are then overwritten by the checkpoint, which holds the full encoder).

DepthSplat's ``render_cuda`` / ``DecoderSplattingCUDA`` are MVSplat's (same dcharatan rasterizer fork, same
call), so the upstream decoder runs on our installed ``diff_gaussian_rasterization``.
"""
from .re10k_family import Re10kFamilyEvaluator


class DepthSplatEvaluator(Re10kFamilyEvaluator):
    name = 'depthsplat'
    gt_context_poses = True
    repo = 'depthsplat'
    # README, "To evalute the large model" (python -m src.main +experiment=re10k ... mode=test)
    hydra_overrides = ['model.encoder.num_scales=2', 'model.encoder.upsample_factor=2',
                       'model.encoder.lowest_feature_resolution=4', 'model.encoder.monodepth_vit_type=vitl']

    def encode(self, inputs):
        out = self.encoder(inputs['context'], 0, deterministic=False)
        # return_depth=True: {'gaussians', 'depths'}; in eval mode only the final-scale depth is used
        return out['gaussians'] if isinstance(out, dict) else out
