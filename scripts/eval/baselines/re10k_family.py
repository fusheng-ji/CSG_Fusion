"""pixelSplat, MVSplat and NoPoSplat, and the base class of the other baselines on pixelSplat's code base.

Each is built exactly like its ``src/main.py`` does in test mode (Hydra config ``+experiment=re10k`` with
the released RealEstate10K checkpoint), and each CSG-Fusion sample is converted the way the baseline's own
``DatasetRE10k`` converts an example: 256x256 images, OpenCV camera-to-world extrinsics, intrinsics
normalised by the image size, optional baseline-1 rescaling (``make_baseline_1``), NoPoSplat's
``relative_pose`` canonicalisation, and the dataset's near / far bounds. The upstream data shim then runs
as in ``ModelWrapper.test_step``.
"""
import torch
import torch.nn.functional as F

from .common import BaselineEvaluator, render_cuda, stack_views


class Re10kFamilyEvaluator(BaselineEvaluator):

    image_shape = (256, 256)           # resolution the RealEstate10K checkpoints are trained at
    experiment = 're10k'
    hydra_overrides = []

    # ---------------------------------------------------------------------------------------------------
    def build(self, overrides):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from src.config import load_typed_root_config
        from src.dataset.data_module import get_data_shim
        from src.dataset.dataset_re10k import DatasetRE10k
        from src.global_cfg import set_cfg
        from src.model.encoder import get_encoder

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(self.sandbox.root / 'config'), version_base=None):
            cfg_dict = compose('main', overrides=[f'+experiment={self.experiment}', 'mode=test', 'wandb.mode=disabled',
                                                  *self.hydra_overrides, *overrides])
        set_cfg(cfg_dict)
        cfg = load_typed_root_config(cfg_dict)
        self.cfg = cfg

        encoder, _ = get_encoder(cfg.model.encoder)
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        state = state.get('state_dict', state)
        encoder.load_state_dict({k[len('encoder.'):]: v for k, v in state.items() if k.startswith('encoder.')})
        self.encoder = encoder.to(self.device).eval()
        self.decoder = self.make_decoder(cfg).to(self.device).eval()
        self.data_shim = get_data_shim(self.encoder)

        dcfg = self.dataset_cfg(cfg)
        self.make_baseline_1 = dcfg.make_baseline_1
        self.relative_pose = getattr(dcfg, 'relative_pose', False)
        self.baseline_scale_bounds = getattr(dcfg, 'baseline_scale_bounds', True)
        # DatasetRE10k.get_bound reads self.near / self.far: class defaults, overridden by the cfg in MVSplat.
        self.near = dcfg.near if getattr(dcfg, 'near', -1) not in (-1, None) else DatasetRE10k.near
        self.far = dcfg.far if getattr(dcfg, 'far', -1) not in (-1, None) else DatasetRE10k.far

    def make_decoder(self, cfg):
        from src.model.decoder import get_decoder
        return get_decoder(cfg.model.decoder, cfg.dataset)

    def dataset_cfg(self, cfg):
        return cfg.dataset

    def modules(self):
        return [self.encoder]

    # ---------------------------------------------------------------------------------------------------
    def prepare(self, batch):
        ctx, tgt = batch['context'], batch['target']
        h0, w0 = ctx[0]['original_img'].shape[-2:]
        b = ctx[0]['original_img'].shape[0]

        def images(views):
            x = stack_views(views, 'original_img')                                   # (b, v, 3, H, W) in [0, 1]
            v = x.shape[1]
            x = F.interpolate(x.flatten(0, 1), size=self.image_shape, mode='bilinear', align_corners=False,
                              antialias=True)
            return x.unflatten(0, (b, v)).clamp(0, 1)

        def intrinsics(views):
            k = stack_views(views, 'camera_intrinsics')[..., :3, :3].float().clone()
            k[..., 0, :] /= w0
            k[..., 1, :] /= h0
            return k

        extrinsics = torch.cat([stack_views(ctx, 'camera_pose'), stack_views(tgt, 'camera_pose')], 1).float().clone()
        n_ctx = len(ctx)

        # DatasetRE10k: "Resize the world to make the baseline 1."
        scale = torch.ones(b, device=extrinsics.device)
        if self.make_baseline_1:
            scale = (extrinsics[:, 0, :3, 3] - extrinsics[:, n_ctx - 1, :3, 3]).norm(dim=-1)
            extrinsics[..., :3, 3] /= scale[:, None, None]
        if self.relative_pose:
            from src.misc.cam_utils import camera_normalization
            extrinsics = torch.stack([camera_normalization(e[0:1], e) for e in extrinsics])
        nf_scale = scale if self.baseline_scale_bounds else torch.ones_like(scale)

        def bounds(value, v):
            return (value / nf_scale)[:, None].expand(b, v).contiguous()

        def views(name, views_, e):
            v = len(views_)
            return {'extrinsics': e, 'intrinsics': intrinsics(views_), 'image': images(views_),
                    'near': bounds(self.near, v), 'far': bounds(self.far, v),
                    'index': torch.tensor([[int(x['instance'][i]) if str(x['instance'][i]).isdigit() else 0
                                            for x in views_] for i in range(b)], device=e.device)}

        example = {'context': views('context', ctx, extrinsics[:, :n_ctx]),
                   'target': views('target', tgt, extrinsics[:, n_ctx:]),
                   'scene': batch['scene']}
        example['context']['overlap'] = torch.zeros(b, device=extrinsics.device)
        return self.data_shim(example)

    def encode(self, inputs):
        return self.encoder(inputs['context'], 0, deterministic=False)

    # --- training-free fusion adapters (src/fusion_plugin.py): Gaussians are 'b (v r srf spp) ...',
    # i.e. view-major, rendered in the context-camera frame given by inputs['context']['extrinsics'].
    def fusion_views(self, gaussians, inputs):
        n = gaussians.means.shape[1]
        n_ctx = inputs['context']['extrinsics'].shape[1]
        h, w = inputs['context']['image'].shape[-2:]
        K = inputs['context']['intrinsics'][0, 0].clone()
        K[0] *= w
        K[1] *= h
        return dict(means=gaussians.means[0], opacities=gaussians.opacities[0], n1=n // n_ctx,
                    c2w1=inputs['context']['extrinsics'][0, 0], K1=K, hw=(h, w))

    def fusion_drop(self, gaussians, keep, opacities=None):
        import dataclasses
        n = keep.shape[0]
        if opacities is not None:
            gaussians = dataclasses.replace(gaussians, opacities=opacities.reshape(1, -1).to(gaussians.opacities))
        return dataclasses.replace(gaussians, **{
            f.name: getattr(gaussians, f.name)[:, keep] for f in dataclasses.fields(gaussians)
            if torch.is_tensor(getattr(gaussians, f.name)) and getattr(gaussians, f.name).shape[1:2] == (n,)})

    def render(self, gaussians, inputs, views):
        d = inputs[views]
        h, w = d['image'].shape[-2:]
        out = self.decoder.forward(gaussians, d['extrinsics'], d['intrinsics'], d['near'], d['far'], (h, w))
        return out.color


class PixelSplatEvaluator(Re10kFamilyEvaluator):
    """pixelSplat (Charatan et al., CVPR 2024), https://github.com/dcharatan/pixelsplat"""
    name = 'pixelsplat'
    gt_context_poses = True
    repo = 'pixelsplat'


class MVSplatEvaluator(Re10kFamilyEvaluator):
    """MVSplat (Chen et al., ECCV 2024), https://github.com/donydchen/mvsplat"""
    name = 'mvsplat'
    gt_context_poses = True
    repo = 'mvsplat'


class NoPoSplatEvaluator(Re10kFamilyEvaluator):
    """NoPoSplat (Ye et al., ICLR 2025), https://github.com/cvg/NoPoSplat

    NoPoSplat's decoder calls the pose-optimising rasterizer fork (rmurai0610/diff-gaussian-rasterization-w-pose),
    which installs under the same package name as ours but returns five outputs. Its colour pass is the same
    3DGS forward, so the Gaussians are rendered with CSG-Fusion's ``render_cuda`` (the pixelSplat renderer,
    same projection / SH / scale-invariant handling as NoPoSplat's ``render_cuda`` without the pose deltas).
    """
    name = 'noposplat'
    repo = 'NoPoSplat'

    def make_decoder(self, cfg):
        from src.model.decoder import get_decoder
        return get_decoder(cfg.model.decoder)

    def dataset_cfg(self, cfg):
        wrapper = cfg.dataset[0]                       # list[DatasetCfgWrapper]; the wrapper holds one dataset cfg
        return next(iter(vars(wrapper).values()))

    def encode(self, inputs):
        return self.encoder(inputs['context'], 0)

    def render(self, gaussians, inputs, views):
        d = inputs[views]
        b, v = d['extrinsics'].shape[:2]
        h, w = d['image'].shape[-2:]
        rep = lambda x: x[:, None].expand(b, v, *x.shape[1:]).flatten(0, 1)
        color = render_cuda(
            d['extrinsics'].flatten(0, 1), d['intrinsics'].flatten(0, 1), d['near'].flatten(), d['far'].flatten(),
            (h, w), self.decoder.background_color[None].expand(b * v, 3),
            rep(gaussians.means), rep(gaussians.covariances), rep(gaussians.harmonics), rep(gaussians.opacities),
            scale_invariant=self.decoder.make_scale_invariant)
        return color.unflatten(0, (b, v))
