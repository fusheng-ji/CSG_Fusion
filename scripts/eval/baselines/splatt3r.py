"""Splatt3R (Smart et al., 2024), https://github.com/btsmart/splatt3r

CSG-Fusion's data pipeline is Splatt3R's, so a CSG-Fusion test batch is already a Splatt3R batch: the model
gets the same 512x512 inputs as CSG-Fusion and renders at 512x512. Context views are rendered by handing
the upstream decoder a batch whose targets are the context views.
"""
import torch

from .common import BaselineEvaluator


class Splatt3REvaluator(BaselineEvaluator):
    name = 'splatt3r'
    repo = 'splatt3r'

    def build(self, overrides):
        import main  # Splatt3R's main.py (inside the sandbox)
        self.model = main.MAST3RGaussians.load_from_checkpoint(str(self.checkpoint), map_location='cpu')
        self.model = self.model.to(self.device).eval()

    def prepare(self, batch):
        return batch

    def encode(self, inputs):
        view1, view2 = inputs['context']
        return self.model(view1, view2)

    def count_gaussians(self, out):
        return sum(p['opacities'][0].numel() for p in out)       # one per pixel and view

    def render(self, gaussians, inputs, views):
        h, w = inputs['context'][0]['img'].shape[-2:]
        if isinstance(gaussians, dict):                      # after fusion (flat Gaussians)
            return self.render_flat(gaussians, inputs, views, h, w)
        pred1, pred2 = gaussians
        color, _ = self.model.decoder(dict(inputs, target=inputs[views]), pred1, pred2, (h, w))
        return color

    # --- training-free fusion adapters (src/fusion_plugin.py) -------------------------------------------
    # Splatt3R's Gaussians live in view 1's camera frame (its decoder maps the target poses there), one per
    # pixel per view. For fusion they are flattened ('flat': view-1 pixels first), and rendered with the same
    # render_cuda call as the upstream decoder, so dropped Gaussians are physically removed.
    def _flat(self, gaussians):
        if isinstance(gaussians, dict):
            return gaussians
        from einops import rearrange
        pred1, pred2 = gaussians
        cat = lambda a, b, pat: torch.cat([rearrange(a, pat), rearrange(b, pat)], dim=1)
        return {'means': cat(pred1['means'], pred2['means_in_other_view'], 'b h w x -> b (h w) x'),
                'covariances': cat(pred1['covariances'], pred2['covariances'], 'b h w i j -> b (h w) i j'),
                'sh': cat(pred1['sh'], pred2['sh'], 'b h w c d -> b (h w) c d'),
                'opacities': cat(pred1['opacities'], pred2['opacities'], 'b h w 1 -> b (h w)')}

    def fusion_views(self, gaussians, inputs):
        g = self._flat(gaussians)
        h, w = inputs['context'][0]['img'].shape[-2:]
        return dict(means=g['means'][0], opacities=g['opacities'][0], n1=g['means'].shape[1] // 2,
                    c2w1=torch.eye(4, device=g['means'].device),
                    K1=inputs['context'][0]['camera_intrinsics'][0, :3, :3].float(), hw=(h, w))

    def fusion_drop(self, gaussians, keep, opacities=None):
        g = dict(self._flat(gaussians))
        if opacities is not None:
            g['opacities'] = opacities.reshape(1, -1).to(g['opacities'])
        return {k: x[:, keep] for k, x in g.items()}

    def render_flat(self, g, inputs, views, h, w):
        """The upstream decoder's render_cuda call, on flat Gaussians."""
        from einops import rearrange, repeat
        from src.pixelsplat_src.cuda_splatting import render_cuda as upstream_render_cuda
        from utils.geometry import normalize_intrinsics
        batch = dict(inputs, target=inputs[views])
        base = torch.inverse(batch['context'][0]['camera_pose'])
        extr = base[:, None] @ torch.stack([t['camera_pose'] for t in batch['target']], dim=1)
        intr = normalize_intrinsics(torch.stack([t['camera_intrinsics'] for t in batch['target']], dim=1), (h, w))[..., :3, :3]
        b, v = extr.shape[:2]
        rep = lambda x: repeat(x, 'b g ... -> (b v) g ...', v=v)
        color = upstream_render_cuda(
            rearrange(extr, 'b v i j -> (b v) i j'), rearrange(intr, 'b v i j -> (b v) i j'),
            torch.full((b * v,), 0.1, device=extr.device), torch.full((b * v,), 1000.0, device=extr.device), (h, w),
            repeat(self.model.decoder.background_color, 'c -> n c', n=b * v),
            rep(g['means']), rep(g['covariances']), rep(g['sh']), rep(g['opacities']))
        return rearrange(color, '(b v) c h w -> b v c h w', b=b, v=v)
