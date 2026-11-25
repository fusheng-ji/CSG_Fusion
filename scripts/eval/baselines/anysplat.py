"""AnySplat (Jiang et al., SIGGRAPH Asia 2025), https://github.com/Sage-PhysicalAI/AnySplat

Pose-free and intrinsics-free: the VGGT-based encoder predicts Gaussians *and* the context cameras
(extrinsics + intrinsics) from the context images alone; no ground-truth pose, scale or intrinsics reach the
model. The model is the released Hugging Face checkpoint ``lhjiang/anysplat`` loaded with
``AnySplat.from_pretrained`` (as in ``inference.py`` / ``src/eval_nvs.py``).

Everything mirrors the upstream NVS evaluation ``src/eval_nvs.py``:

* images: ``src/utils/image.process_image`` (PIL resize of the short side to 448 + centre crop to 448x448;
  our 512x512 crops are square, so this is a plain 512 -> 448 PIL resize of the 8-bit image), fed in [0, 1];
* Gaussians: ``model.encoder(context_images)`` -- the target image is *not* used for the Gaussians;
* target camera: the target image *is* fed to the network, but only for its pose. ``eval_nvs.py`` runs the
  aggregator + camera head once more on ``cat(context, target)``, takes the predicted target c2w extrinsics
  and intrinsics, and rescales their translations by
  ``mean(t of the encoder's context poses) / mean(t of this joint pass's context poses)``
  (both passes put the world frame at context view 0, so only the scale differs). We do exactly that.
  No ground-truth pose / scale is used anywhere, so ``gt_context_poses = False``;
* rendering: the upstream ``DecoderSplattingCUDA`` (gsplat ``rasterization``; white background, as in the
  released config). Context views are rendered at the encoder's own predicted context cameras
  (as ``AnySplat.forward`` does); targets at the rescaled predicted target cameras, at 448x448.

Timing: ``encode`` includes the extra target-pose pass (aggregator + camera head on all three images), since
it is network compute needed before the novel view can be rendered; ``render`` is rasterisation only.

Environment adaptations (upstream files untouched): ``torch_scatter`` (only ``scatter_add`` / ``scatter_max``
in the voxelisation) is replaced by an equivalent pure-PyTorch module, and ``xformers`` (imported but unused
on the inference path -- the VGGT layers already use SDPA) by an empty stub; both stubs exist only while the
upstream modules are being imported.
"""
import contextlib
import io
import sys
import types

import numpy as np
import torch
from PIL import Image

from .common import BaselineEvaluator, stack_views


def _torch_scatter_stub():
    """Pure-PyTorch ``torch_scatter.scatter_add`` / ``scatter_max`` for the calls in EncoderAnySplat
    (1-D ``index`` along ``dim=0``; every output slot receives at least one element)."""
    mod = types.ModuleType('torch_scatter')

    def _expand(src, index):
        return index.long().view(-1, *([1] * (src.dim() - 1))).expand_as(src)

    def scatter_add(src, index, dim=0, out=None, dim_size=None):
        assert dim == 0 and out is None
        n = int(index.max()) + 1 if dim_size is None else dim_size
        return torch.zeros((n, *src.shape[1:]), dtype=src.dtype, device=src.device).scatter_add_(
            0, _expand(src, index), src)

    def scatter_max(src, index, dim=0, out=None, dim_size=None):
        assert dim == 0 and out is None
        n = int(index.max()) + 1 if dim_size is None else dim_size
        res = torch.zeros((n, *src.shape[1:]), dtype=src.dtype, device=src.device).scatter_reduce_(
            0, _expand(src, index), src, reduce='amax', include_self=False)
        return res, None                       # EncoderAnySplat only uses the values

    mod.scatter_add, mod.scatter_max = scatter_add, scatter_max
    return mod


@contextlib.contextmanager
def _import_stubs():
    stubs = {'torch_scatter': _torch_scatter_stub()}
    xf, xops = types.ModuleType('xformers'), types.ModuleType('xformers.ops')
    xf.ops = xops
    stubs.update({'xformers': xf, 'xformers.ops': xops})
    added = [k for k in stubs if k not in sys.modules]
    for k in added:
        sys.modules[k] = stubs[k]
    try:
        yield
    finally:
        for k in added:
            sys.modules.pop(k, None)


class AnySplatEvaluator(BaselineEvaluator):
    target_pose_source = 'predicted (target image)'
    name = 'anysplat'
    repo = 'AnySplat'
    gt_context_poses = False
    image_size = 448                     # src/utils/image.process_image

    def build(self, overrides):
        with _import_stubs():
            from src.model.model.anysplat import AnySplat
            from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
        self.pose_encoding_to_extri_intri = pose_encoding_to_extri_intri
        # checkpoint: local snapshot of huggingface.co/lhjiang/anysplat (config.json + model.safetensors)
        path = self.checkpoint if self.checkpoint.is_dir() else self.checkpoint.parent
        with contextlib.redirect_stdout(io.StringIO()):
            model = AnySplat.from_pretrained(str(path), strict=True)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    # ---------------------------------------------------------------------------------------------------
    def _images(self, views):
        """process_image on our 512x512 crop: 8-bit PIL image, PIL resize (default resample) to 448, [0, 1]."""
        x = stack_views(views, 'original_img')                                  # (b, v, 3, H, W) in [0, 1]
        b, v = x.shape[:2]
        out = []
        for img in x.flatten(0, 1):
            a = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            im = Image.fromarray(a)
            w, h = im.size
            if w > h:
                nh, nw = self.image_size, int(w * (self.image_size / h))
            else:
                nw, nh = self.image_size, int(h * (self.image_size / w))
            im = im.resize((nw, nh))
            l, t = (nw - self.image_size) // 2, (nh - self.image_size) // 2
            im = im.crop((l, t, l + self.image_size, t + self.image_size))
            t_ = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255
            out.append(t_ * 2 - 1)                                               # process_image output
        imgs = torch.stack(out).unflatten(0, (b, v)).to(self.device)
        return (imgs + 1) * 0.5                                                  # eval_nvs.py: (x + 1) * 0.5

    def prepare(self, batch):
        return {'context': self._images(batch['context']), 'target': self._images(batch['target'])}

    def reconstruct(self, inputs):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.model.encoder(inputs['context'], global_step=0, visualization_dump={}).gaussians

    def encode(self, inputs):
        model = self.model
        ctx, tgt = inputs['context'], inputs['target']
        b, v_ctx = ctx.shape[:2]
        h, w = tgt.shape[-2:]
        with contextlib.redirect_stdout(io.StringIO()):          # the upstream encoder prints statistics
            out = model.encoder(ctx, global_step=0, visualization_dump={})
        gaussians, pred_context_pose = out.gaussians, out.pred_context_pose
        self.encode_stats = {'num_gaussians': float(gaussians.means.shape[1])}   # after the upstream voxel merge

        # --- src/eval_nvs.py: target pose from a joint pass over context + target images ---
        vggt_input = torch.cat((ctx, tgt), dim=1).to(torch.bfloat16)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
            tokens, _ = model.encoder.aggregator(vggt_input, intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx)
        with torch.cuda.amp.autocast(enabled=False):
            fp32_tokens = [t.float() for t in tokens]
            pose_enc = model.encoder.camera_head(fp32_tokens)[-1]
            all_extr, all_intr = self.pose_encoding_to_extri_intri(pose_enc, vggt_input.shape[-2:])
        pad = torch.tensor([0, 0, 0, 1], device=all_extr.device, dtype=all_extr.dtype).view(1, 1, 1, 4).repeat(
            b, vggt_input.shape[1], 1, 1)
        all_extr = torch.cat([all_extr, pad], dim=2).inverse()                  # c2w
        all_intr = all_intr.clone()
        all_intr[:, :, 0] = all_intr[:, :, 0] / w
        all_intr[:, :, 1] = all_intr[:, :, 1] / h
        ctx_extr, tgt_extr = all_extr[:, :v_ctx], all_extr[:, v_ctx:]
        tgt_intr = all_intr[:, v_ctx:]
        scale_factor = pred_context_pose['extrinsic'][:, :, :3, 3].mean() / ctx_extr[:, :, :3, 3].mean()
        tgt_extr = tgt_extr.clone()
        tgt_extr[..., :3, 3] = tgt_extr[..., :3, 3] * scale_factor
        cams = {'context': (pred_context_pose['extrinsic'], pred_context_pose['intrinsic']),
                'target': (tgt_extr, tgt_intr.float())}
        return gaussians, cams, float(scale_factor)

    def render(self, gaussians, inputs, views):
        gaussians, cams, _ = gaussians
        extr, intr = cams[views]
        b, v = extr.shape[:2]
        h, w = inputs[views].shape[-2:]
        near = torch.ones(b, v, device=self.device) * 0.01
        far = torch.ones(b, v, device=self.device) * 100
        out = self.model.decoder.forward(gaussians, extr, intr, near, far, (h, w))
        return out.color.float().clamp(0, 1)
