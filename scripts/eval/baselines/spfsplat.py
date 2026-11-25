"""SPFSplat (Huang & Mikolajczyk, ICCV 2025) and SPFSplatV2-L (Huang & Mikolajczyk, 2025): pose-free,
self-supervised Gaussian splatting on the NoPoSplat code base.

https://github.com/ranrhuang/SPFSplat   (third_party/baselines/SPFSplat,   MASt3R / CroCo backbone, 256x256)
https://github.com/ranrhuang/SPFSplatV2 (third_party/baselines/SPFSplatV2, SPFSplatV2-L = VGGT backbone, 224x224)

Both are built exactly like ``src/main.py`` builds them in test mode (Hydra ``+experiment=spfsplat/re10k`` resp.
``+experiment=spfsplatv2-l/re10k`` with the released 2-view RealEstate10K checkpoint), and a CSG-Fusion sample is
converted the way their ``DatasetRE10k`` converts one (see ``Re10kFamilyEvaluator.prepare``).

Official NVS protocol (``ModelWrapper.test_step`` with ``estimating_pose=True``, ``test.align_pose=false``, as in
both READMEs): the encoder gets the context images *and the target image* plus the GT intrinsics of all of them
(no GT pose at all). Gaussians and context poses come from the context views only (SPFSplat decodes the context
tokens without the target; SPFSplatV2-L masks the target out of the context tokens' attention); the pose head
additionally predicts the target camera in the canonical frame (context view 1 = identity; SPFSplat: predicted
context baseline = 1, SPFSplatV2-L: VGGT's own scale), and the target is rendered from that predicted pose. The
context views are rendered from their predicted poses (``extrinsics['c']``). Near / far are the dataset's
(0.1 / 100 divided by the GT context baseline, ``make_baseline_1``), as in their data loader.

Deviations (implementation only, no change of the computation):
* Their decoder imports the pose-optimising rasterizer fork ``diff_gauss_pose`` (slothfulxtx), which is only needed
  for pose gradients. The Gaussians are rendered with CSG-Fusion's ``render_cuda`` (same 3DGS forward; we pass the
  covariance, which their adapter builds from the same scales / rotations the fork receives), like NoPoSplat.
* ``src/misc/cam_utils.py`` imports ``pytorch3d.transforms`` at module level; inference only uses
  ``rotation_6d_to_matrix`` (SPFSplat's pose head). pytorch3d is not installed (CUDA build), so a stub module with
  pytorch3d's exact implementation of that function is registered (only if pytorch3d is not importable); likewise
  a ``diff_gauss_pose`` stub that raises if called, since ``src.loss`` imports the decoder module at import time.
* SPFSplatV2-L's backbone would download facebook/VGGT-1B at construction; we build it with
  ``model.encoder.backbone.pretrained=false`` and load all weights from the released checkpoint instead.
"""
import sys
import types

import torch
import torch.nn.functional as F

from .common import render_cuda
from .re10k_family import Re10kFamilyEvaluator


def _rotation_6d_to_matrix(d6):
    """pytorch3d.transforms.rotation_6d_to_matrix (pytorch3d/transforms/rotation_conversions.py), verbatim."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class _NoPoseRasterizer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('diff_gauss_pose is not installed: render with render_cuda (see module docstring)')


def _install_diff_gauss_pose_stub():
    """Their decoder module imports ``diff_gauss_pose`` at module level (via src.loss -> src.model.decoder); we never
    call it. Registered only if the real fork is not importable."""
    try:
        import diff_gauss_pose  # noqa: F401
        return
    except ImportError:
        pass
    m = types.ModuleType('diff_gauss_pose')
    m.GaussianRasterizationSettings = m.GaussianRasterizer = _NoPoseRasterizer
    sys.modules['diff_gauss_pose'] = m


def _install_pytorch3d_stub():
    if 'pytorch3d.transforms' in sys.modules:
        return
    try:
        import pytorch3d.transforms  # noqa: F401  (a real install wins)
        return
    except ImportError:
        pass
    pkg = types.ModuleType('pytorch3d')
    pkg.__path__ = []
    tr = types.ModuleType('pytorch3d.transforms')
    tr.rotation_6d_to_matrix = _rotation_6d_to_matrix
    pkg.transforms = tr
    sys.modules['pytorch3d'] = pkg
    sys.modules['pytorch3d.transforms'] = tr


class _SPFSplatBase(Re10kFamilyEvaluator):

    gt_context_poses = False
    target_pose_source = 'predicted (target image)'

    def build(self, overrides):
        _install_pytorch3d_stub()
        _install_diff_gauss_pose_stub()
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from src.config import load_typed_root_config
        from src.dataset.dataset_re10k import DatasetRE10k
        from src.global_cfg import set_cfg
        from src.model.encoder import get_encoder

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(self.sandbox.root / 'config'), version_base=None):
            cfg_dict = compose('main', overrides=[f'+experiment={self.experiment}', 'mode=test', 'wandb.mode=disabled',
                                                  'test.align_pose=false', *self.hydra_overrides, *overrides])
        set_cfg(cfg_dict)
        cfg = load_typed_root_config(cfg_dict)
        self.cfg = cfg
        assert cfg.model.encoder.estimating_pose and not cfg.model.encoder.estimating_focal

        encoder, _ = get_encoder(cfg.model.encoder)
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        state = state.get('state_dict', state)
        encoder.load_state_dict({k[len('encoder.'):]: v for k, v in state.items() if k.startswith('encoder.')})
        del state
        self.encoder = encoder.to(self.device).eval()
        self.data_shim = lambda batch: batch      # their ModelWrapper builds the shim but never applies it

        dec = cfg.model.decoder
        self.background_color = torch.tensor(dec.background_color, dtype=torch.float32, device=self.device)
        self.make_scale_invariant = dec.make_scale_invariant

        dcfg = self.dataset_cfg(cfg)
        self.image_shape = tuple(dcfg.input_image_shape)
        self.make_baseline_1 = dcfg.make_baseline_1
        self.relative_pose = dcfg.relative_pose
        self.baseline_scale_bounds = getattr(dcfg, 'baseline_scale_bounds', True)
        self.near = dcfg.near if dcfg.near != -1 else DatasetRE10k.near
        self.far = dcfg.far if dcfg.far != -1 else DatasetRE10k.far

    def dataset_cfg(self, cfg):
        wrapper = cfg.dataset[0]
        return next(iter(vars(wrapper).values()))

    # prepare: Re10kFamilyEvaluator.prepare. The GT extrinsics it computes are never given to the encoder (only
    # images + intrinsics).

    def reconstruct(self, inputs):
        return self.encoder(inputs['context'], 0, visualization_dump=None, target=None)['gaussians']

    def encode(self, inputs):
        ctx, tgt = inputs['context'], inputs['target']
        target = {k: tgt[k] for k in ('image', 'intrinsics', 'near', 'far')}
        # test_step loops over target views one at a time; CSG test samples have one target view
        assert tgt['image'].shape[1] == 1
        out = self.encoder(ctx, 0, visualization_dump=None, target=target)
        v_cxt = ctx['image'].shape[1]
        ext = {'context': out['extrinsics']['c'][:, :v_cxt],
               'target': out['extrinsics']['cwt'][:, v_cxt:]}
        return {'gaussians': out['gaussians'], 'extrinsics': ext}

    def render(self, output, inputs, views):
        g, e = output['gaussians'], output['extrinsics'][views]
        d = inputs[views]
        b, v = e.shape[:2]
        h, w = d['image'].shape[-2:]
        rep = lambda x: x[:, None].expand(b, v, *x.shape[1:]).flatten(0, 1)
        color = render_cuda(
            e.flatten(0, 1).float(), d['intrinsics'].flatten(0, 1), d['near'].flatten(), d['far'].flatten(), (h, w),
            self.background_color[None].expand(b * v, 3),
            rep(g.means), rep(g.covariances), rep(g.harmonics), rep(g.opacities),
            scale_invariant=self.make_scale_invariant)
        return color.unflatten(0, (b, v))


class SPFSplatEvaluator(_SPFSplatBase):
    """SPFSplat (ICCV 2025), re10k.ckpt: 2 views, 256x256, intrinsics embedding (GT intrinsics as input)."""
    name = 'spfsplat'
    repo = 'SPFSplat'
    experiment = 'spfsplat/re10k'


class SPFSplatV2LEvaluator(_SPFSplatBase):
    """SPFSplatV2-L (VGGT backbone), re10k_spfsplatv2l.ckpt: 2 views, 224x224, GT intrinsics embedded in the decoder."""
    name = 'spfsplatv2_l'
    repo = 'SPFSplatV2'
    experiment = 'spfsplatv2-l/re10k'
    hydra_overrides = ['model.encoder.backbone.pretrained=false']
