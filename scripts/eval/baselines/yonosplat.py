"""YoNoSplat (Ye et al., ICLR 2026), https://github.com/cvg/YoNoSplat

Evaluated in its main pose-free NVS setting with ground-truth intrinsics ("Predicted pose, ground-truth
intrinsics" in EVALUATION.md), with the released RealEstate10K checkpoint ``re10k_224x224_ctx2to32.ckpt``
and two context views:

    +experiment=yono_re10k mode=test model.encoder.pose_free=true test.align_pose=true
    model.decoder.prune_opacity_threshold=0.005 dataset.re10k.view_sampler.num_context_views=2

Each CSG-Fusion sample is converted the way ``DatasetRE10k`` + ``apply_crop_shim`` convert an example:

* images: the 512x512 crop is quantised to uint8 and resized to 224x224 with PIL LANCZOS (upstream
  ``rescale``; for a square input the centre crop is a no-op and the normalised intrinsics are unchanged);
* images go to the model in [0, 1] (the encoder's normalise shim uses mean 0 / std 1);
* intrinsics: GT, normalised by the image size (fed to the backbone's pixel-wise intrinsics embedding);
* poses: OpenCV c2w, translations divided by ``compute_pose_norm_scale(context, 'max_pairwise_d')`` (the
  context baseline for two views), then ``camera_normalization`` to the first context view; near / far =
  0.1 / 100 divided by the same scale. The GT *context* poses are only used for this normalisation of the
  target pose, never fed to the model (``pose_free=true``: the Gaussians are placed with the predicted poses,
  in the frame of the first view, at the scale the model was trained to predict, i.e. context baseline 1).

Target view: exactly ``ModelWrapper.test_step_align``. Starting from the normalised GT target pose, the
target camera is refined for ``test.pose_align_steps`` (100) Adam steps (lr 0.005 for rotation and
translation) on the upstream test-time loss (MSE + 0.05 VGG-LPIPS + opacity, against the GT target image at
224x224), with the Gaussians frozen; then the target is rendered from the aligned pose. Context views are
rendered from the model's own predicted context poses (what the Gaussians were placed with).

Rendering: YoNoSplat's ``splatting_cuda`` decoder calls the pose-optimising rasterizer fork
(rmurai0610/diff-gaussian-rasterization-w-pose), which clashes with ours (see NoPoSplat). The Gaussians are
rendered with CSG-Fusion's ``render_cuda`` instead (same 3DGS forward, same scale-invariant handling). For the
pose alignment, the fork's camera-delta gradients are replaced by autograd through the equivalent rigid
transform of the Gaussians: rendering camera ``c2w`` perturbed by ``w2c' = SE3_exp(tau) @ w2c`` is the same as
rendering the unperturbed camera on Gaussians moved by ``M = c2w @ SE3_exp(tau) @ w2c`` (means ``M x``,
covariances ``R_M S R_M^T``; exact because the checkpoint uses SH degree 0, i.e. view-independent colour).
The pose update itself is upstream ``update_pose``.

Timing: ``decoder_s`` is rasterisation only (target + context renders); the test-time target-pose alignment
(100 optimisation steps) is timed separately and returned by ``predict`` as ``align_s`` (``last_align_s``).

Known limitation: the ScanNet++ 512x512 crops (normalised fx ~0.54, ~85 deg horizontal FOV) are wider than
anything in RealEstate10K (the loader drops FOV > 100 deg; typical ~50-65 deg). The checkpoint's local-point
head then behaves as if fx ~0.87 (its intrinsics head predicts ~0.87 on every sample), so even the context
views re-render at only ~16 dB with GT intrinsics; on a centre crop (fx ~0.71 / 1.07) they re-render at ~30 /
~32 dB, i.e. the harness itself is consistent.
"""
import time

import torch
from torch import nn

from .common import BaselineEvaluator, render_cuda, stack_views


class YoNoSplatEvaluator(BaselineEvaluator):
    target_pose_source = 'gt + alignment to target image'
    name = 'yonosplat'
    repo = 'YoNoSplat'
    gt_context_poses = False

    image_shape = (224, 224)
    experiment = 'yono_re10k'
    hydra_overrides = [
        'model.encoder.pose_free=true',
        'model.encoder.backbone.use_pred_intrinsics_for_embed=false',     # GT intrinsics
        'test.align_pose=true',
        'model.decoder.prune_opacity_threshold=0.005',
        'dataset/view_sampler@dataset.re10k.view_sampler=evaluation',
        'dataset.re10k.view_sampler.num_context_views=2',
    ]

    # ---------------------------------------------------------------------------------------------------
    def build(self, overrides):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from src.config import load_typed_root_config
        from src.dataset.data_module import get_data_shim
        from src.dataset.dataset_re10k import DatasetRE10k
        from src.global_cfg import set_cfg
        from src.loss import get_losses
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
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.data_shim = get_data_shim(self.encoder)
        # test-time pose-alignment loss: the same losses ModelWrapper.test_step_align sums
        self.losses = nn.ModuleList(get_losses(cfg.loss)).to(self.device).eval()

        dcfg = next(iter(vars(cfg.dataset[0]).values()))     # list[DatasetCfgWrapper] -> the re10k cfg
        self.pose_norm_method = dcfg.pose_norm_method
        self.relative_pose = dcfg.relative_pose
        self.near, self.far = DatasetRE10k.near, DatasetRE10k.far
        assert tuple(dcfg.input_image_shape) == self.image_shape

        dec = cfg.model.decoder
        self.background = torch.tensor(dec.background_color, dtype=torch.float32, device=self.device)
        self.scale_invariant = dec.make_scale_invariant
        self.prune_threshold = dec.prune_opacity_threshold
        self.test_cfg = cfg.test
        self.last_align_s = 0.0

    def modules(self):
        return [self.encoder]

    def predict(self, batch):
        self.last_align_s = 0.0
        out = super().predict(batch)
        out['decoder_s'] -= self.last_align_s       # keep decoder_s = rasterisation only
        out['align_s'] = self.last_align_s
        return out

    # ---------------------------------------------------------------------------------------------------
    def prepare(self, batch):
        from src.dataset.norm_scale import compute_pose_norm_scale
        from src.dataset.shims.crop_shim import rescale
        from src.misc.cam_utils import camera_normalization

        ctx, tgt = batch['context'], batch['target']
        h0, w0 = ctx[0]['original_img'].shape[-2:]
        assert h0 == w0, 'square CSG crop expected (rescale_and_crop is then a plain resize)'

        def images(views):
            x = stack_views(views, 'original_img').float()                          # (b, v, 3, H, W) in [0, 1]
            return torch.stack([torch.stack([rescale(im, self.image_shape) for im in xb]) for xb in x])

        def intrinsics(views):
            k = stack_views(views, 'camera_intrinsics')[..., :3, :3].float().clone()
            k[..., 0, :] /= w0
            k[..., 1, :] /= h0
            return k

        n_ctx = len(ctx)
        extrinsics = torch.cat([stack_views(ctx, 'camera_pose'), stack_views(tgt, 'camera_pose')], 1).float().clone()
        scales = []
        for i in range(extrinsics.shape[0]):
            scale = compute_pose_norm_scale(extrinsics[i, :n_ctx], self.pose_norm_method)
            scale = torch.as_tensor(scale, dtype=torch.float32, device=extrinsics.device)
            extrinsics[i, :, :3, 3] /= scale
            if self.relative_pose:
                extrinsics[i] = camera_normalization(extrinsics[i, 0:1], extrinsics[i])
            scales.append(scale)
        scale = torch.stack(scales)
        b = extrinsics.shape[0]

        def views(views_, e):
            v = len(views_)
            return {'extrinsics': e, 'intrinsics': intrinsics(views_), 'image': images(views_),
                    'near': (self.near / scale)[:, None].expand(b, v).contiguous(),
                    'far': (self.far / scale)[:, None].expand(b, v).contiguous()}

        example = {'context': views(ctx, extrinsics[:, :n_ctx]), 'target': views(tgt, extrinsics[:, n_ctx:]),
                   'scene': batch['scene']}
        return self.data_shim(example)          # apply_normalize_shim with mean 0 / std 1: images stay in [0, 1]

    def count_gaussians(self, out):
        return int(out['gaussians'].means.shape[1])

    def encode(self, inputs):
        from src.model.decoder.prune import prune_gaussians
        dump = {}
        gaussians = self.encoder(inputs['context'], 0, visualization_dump=dump)
        # DecoderSplattingCUDA.forward prunes (opacity > 0.005) before every rasterisation; do it once here
        gaussians = prune_gaussians(gaussians, self.prune_threshold, 0.0, 0.0, inference=True)
        return {'gaussians': gaussians, 'context_c2w': dump['c2w'].float()}

    # ---------------------------------------------------------------------------------------------------
    def _rasterize(self, gaussians, extrinsics, intrinsics, near, far, shape, cam_rot_delta=None,
                   cam_trans_delta=None):
        """DecoderSplattingCUDA.forward with CSG's render_cuda; (b, v, ...) cameras -> (b, v, 3, h, w)."""
        b, v = extrinsics.shape[:2]
        rep = lambda x: x[:, None].expand(b, v, *x.shape[1:]).flatten(0, 1)
        means, covs = rep(gaussians.means), rep(gaussians.covariances)
        e = extrinsics.flatten(0, 1)
        if cam_rot_delta is not None:
            from src.misc.cam_utils import SE3_exp
            tau = torch.cat([cam_trans_delta, cam_rot_delta], -1).flatten(0, 1)
            delta = torch.stack([SE3_exp(t) for t in tau])                         # w2c' = delta @ w2c
            m = e @ delta @ e.inverse()                                            # equivalent Gaussian motion
            r, t = m[:, :3, :3], m[:, :3, 3]
            means = means @ r.transpose(1, 2) + t[:, None]
            covs = r[:, None] @ covs @ r[:, None].transpose(-1, -2)
        color = render_cuda(e, intrinsics.flatten(0, 1), near.flatten(), far.flatten(), shape,
                            self.background[None].expand(b * v, 3), means, covs, rep(gaussians.harmonics),
                            rep(gaussians.opacities), scale_invariant=self.scale_invariant)
        return color.unflatten(0, (b, v))

    def _align_target(self, gaussians, d):
        """ModelWrapper.test_step_align: photometric refinement of the target pose, Gaussians frozen."""
        from einops import rearrange
        from src.misc.cam_utils import update_pose
        from src.model.decoder.decoder import DecoderOutput

        b, v, _, h, w = d['image'].shape
        batch = {'target': d}
        with torch.enable_grad():
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            pose_optimizer = torch.optim.Adam([{'params': [cam_rot_delta], 'lr': self.test_cfg.rot_opt_lr},
                                               {'params': [cam_trans_delta], 'lr': self.test_cfg.trans_opt_lr}])
            extrinsics = d['extrinsics'].clone()
            for _ in range(self.test_cfg.pose_align_steps):
                pose_optimizer.zero_grad()
                color = self._rasterize(gaussians, extrinsics, d['intrinsics'], d['near'], d['far'], (h, w),
                                        cam_rot_delta=cam_rot_delta, cam_trans_delta=cam_trans_delta)
                output = DecoderOutput(color, None)
                total_loss = 0
                for loss_fn in self.losses:
                    total_loss = total_loss + loss_fn.forward(output, batch, gaussians, 0)
                total_loss.backward()
                with torch.no_grad():
                    pose_optimizer.step()
                    new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, 'b v i -> (b v) i'),
                                                cam_trans_delta=rearrange(cam_trans_delta, 'b v i -> (b v) i'),
                                                extrinsics=rearrange(extrinsics, 'b v i j -> (b v) i j'))
                    cam_rot_delta.data.fill_(0)
                    cam_trans_delta.data.fill_(0)
                    extrinsics = rearrange(new_extrinsic, '(b v) i j -> b v i j', b=b, v=v)
        return extrinsics.detach()

    def render(self, out, inputs, views):
        gaussians = out['gaussians']
        d = inputs[views]
        h, w = d['image'].shape[-2:]
        if views == 'target':
            extrinsics = d['extrinsics']
            if self.test_cfg.align_pose:
                # test-time pose optimisation; its time is taken out of decoder_s in predict()
                torch.cuda.synchronize(self.device)
                t0 = time.perf_counter()
                extrinsics = self._align_target(gaussians, d)
                torch.cuda.synchronize(self.device)
                self.last_align_s = time.perf_counter() - t0
        else:
            extrinsics = out['context_c2w']             # the model's predicted context poses (first view = I)
        return self._rasterize(gaussians, extrinsics, d['intrinsics'], d['near'], d['far'], (h, w))
