"""Shared evaluation protocol for the baselines.

Every baseline runs from its unmodified upstream repository (``third_party/baselines/<repo>``, a pinned git
submodule). A ``BaselineEvaluator`` subclass only knows how to (1) build the upstream model from its own
config + checkpoint, (2) turn one CSG-Fusion test batch into the baseline's input format, (3) predict
Gaussians and (4) render them into given cameras. Everything that makes the numbers comparable lives here
and is identical for all methods, including CSG-Fusion itself:

* the test samples: ``data/scannetpp/test_set_<a>_<b>.json`` (every 10th sample, as for CSG-Fusion) or the
  DTU index in ``data/dtu_index/``, loaded by CSG-Fusion's own dataset code;
* the ground truth: the 512x512 target view, and on ScanNet++ the depth-consistency mask of
  ``utils/loss_mask.calculate_loss_mask`` (pixels of the target that are visible in a context view);
* the metrics: the test path of ``main.CSGFusion.calculate_loss`` (``apply_mask=True``,
  ``average_over_mask=False``): mask both images, then PSNR / SSIM / LPIPS at 512x512 and, after
  bilinear resizing, at 256x256. Baselines that predict 256x256 images are bilinearly upsampled to 512x512
  first, so the 256x256 numbers go through exactly the same resize as CSG-Fusion's;
* timing (CUDA-synchronised encoder / decoder wall time) and peak GPU memory per sample.
"""
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]                     # repository root
BASELINES_DIR = ROOT / 'third_party' / 'baselines'
for _p in (ROOT, ROOT / 'src' / 'pixelsplat_src', ROOT / 'src' / 'mast3r_src', ROOT / 'src' / 'mast3r_src' / 'dust3r'):
    if str(_p) not in sys.path:
        sys.path.insert(0 if _p == ROOT else len(sys.path), str(_p))     # as main.py does, with absolute paths

# CSG-Fusion modules, imported before any baseline sandbox is entered (see sandbox.py).
_cwd = os.getcwd()
os.chdir(ROOT)
import lpips  # noqa: E402
import data.dtu as dtu  # noqa: E402
import data.scannetpp.scannetpp as scannetpp  # noqa: E402
import utils.compute_ssim as compute_ssim  # noqa: E402
import utils.loss_mask as loss_mask  # noqa: E402
from src.pixelsplat_src.cuda_splatting import render_cuda  # noqa: E402, F401  (re-exported to the evaluators)
from src.fusion_plugin import merge_opacities, random_mask, redundant_mask  # noqa: E402
os.chdir(_cwd)

from .sandbox import ModuleSandbox  # noqa: E402

RESOLUTION = (512, 512)            # CSG-Fusion test resolution (ground truth, mask, Splatt3R / CSG-Fusion renders)


# ======================================================================================================
# Baseline interface
# ======================================================================================================

class BaselineEvaluator:
    """Wraps one upstream baseline. Subclasses implement ``build``, ``prepare``, ``encode`` and ``render``;
    each of them runs inside the baseline's module sandbox."""

    name = None          # method name used on the command line and in result files
    gt_context_poses = False   # True: the model takes the ground-truth context camera poses as input
    target_pose_source = 'gt'  # how the target camera is obtained, as in the method's own NVS evaluation:
                               # 'gt' | 'predicted (target image)' | 'gt + alignment to target image'
    repo = None          # folder under third_party/baselines/

    def __init__(self, checkpoint, device='cuda', overrides=()):
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.exists():
            raise FileNotFoundError(f'{self.name}: checkpoint {self.checkpoint} not found')
        # repo None: the method is CSG-Fusion itself, whose modules are the native ones (no sandbox)
        self.sandbox = ModuleSandbox(BASELINES_DIR / self.repo, hide=[ROOT]) if self.repo else contextlib.nullcontext()
        with self.sandbox:
            self.build(list(overrides))

    # --- to implement ---------------------------------------------------------------------------------
    def build(self, overrides):
        """Create the model from the upstream config and load ``self.checkpoint``."""
        raise NotImplementedError

    def prepare(self, batch):
        """CSG-Fusion test batch (on device) -> the baseline's own input."""
        raise NotImplementedError

    def encode(self, inputs):
        """Baseline input -> Gaussians (in whatever form ``render`` takes)."""
        raise NotImplementedError

    def reconstruct(self, inputs):
        """The context images -> Gaussians, without anything that only serves to place the target camera
        (scripts/eval/bench_efficiency.py). Methods whose ``encode`` also predicts that camera override it."""
        return self.encode(inputs)

    def count_gaussians(self, out):
        """Number of Gaussians in the output of ``reconstruct`` (batch item 0), as passed to the renderer."""
        return int(out.means.shape[1])

    def render(self, gaussians, inputs, views):
        """Render ``views`` ('target' or 'context'); returns (b, v, 3, h, w) in [0, 1]."""
        raise NotImplementedError

    # --- optional: training-free cross-view fusion (src/fusion_plugin.py) ------------------------------
    fusion_mode = 'none'       # 'none' | 'depth' | 'merge' | 'random' (set by the CLI: --fusion, --fusion-tau)
    fusion_tau = 0.01
    fusion_ratio = None        # 'random': fraction of view-2 Gaussians to drop

    def fusion_views(self, gaussians, inputs):
        """-> dict(means (N, 3), opacities (N,), n1 (#view-1 Gaussians, which come first), c2w1 (4, 4),
        K1 (3, 3) pixels, hw) for batch item 0, in the frame the Gaussians are rendered in."""
        raise NotImplementedError(f'{self.name}: no fusion adapter')

    def fusion_drop(self, gaussians, keep, opacities=None):
        """Gaussians with only ``keep`` (N,) bool retained (physically removed, not just transparent); with
        ``opacities`` (N,), those replace the opacities first."""
        raise NotImplementedError(f'{self.name}: no fusion adapter')

    def apply_fusion(self, gaussians, inputs):
        v = self.fusion_views(gaussians, inputs)
        means, n1 = v['means'], v['n1']
        n = means.shape[0]
        opacities = None
        if self.fusion_mode in ('depth', 'merge'):
            red2, partner = redundant_mask(means[n1:], means[:n1], v['c2w1'], v['K1'], v['hw'], self.fusion_tau,
                                           opacities1=v['opacities'][:n1], return_partner=True)
            if self.fusion_mode == 'merge':
                op = v['opacities']
                opacities = torch.cat([merge_opacities(op[:n1], op[n1:], red2, partner), op[n1:]])
        elif self.fusion_mode == 'random':
            g = torch.Generator().manual_seed(int(means.shape[0]))
            red2 = random_mask(n - n1, round(self.fusion_ratio * (n - n1)), generator=g, device=means.device)
        else:
            raise ValueError(self.fusion_mode)
        keep = torch.cat([torch.ones(n1, dtype=torch.bool, device=means.device), ~red2])
        return self.fusion_drop(gaussians, keep, opacities), {'num_gaussians_before': float(n), 'num_gaussians': float(keep.sum()),
                                                   'view2_dropped': float(red2.float().mean())}

    # --- shared -----------------------------------------------------------------------------------------
    def modules(self):
        """The torch modules that make up the model (for the parameter count)."""
        return [self.model]

    def num_parameters(self):
        return sum(p.numel() for m in self.modules() for p in m.parameters())

    @torch.no_grad()
    def predict(self, batch):
        """Returns target / context renders at CSG-Fusion resolution plus encoder / decoder seconds."""
        with self.sandbox:
            inputs = self.prepare(batch)
            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            self.encode_stats = {}
            gaussians = self.encode(inputs)
            torch.cuda.synchronize(self.device)
            t1 = time.perf_counter()
            stats = dict(self.encode_stats)   # e.g. Gaussian counts reported by encode()
            if self.fusion_mode != 'none':
                gaussians, stats = self.apply_fusion(gaussians, inputs)
                torch.cuda.synchronize(self.device)
                stats['fusion_s'] = time.perf_counter() - t1
                t1 = time.perf_counter()
            target = self.render(gaussians, inputs, 'target')
            torch.cuda.synchronize(self.device)
            t2 = time.perf_counter()
            context = self.render(gaussians, inputs, 'context')
        return {'target': to_resolution(target), 'context': to_resolution(context),
                'encoder_s': t1 - t0, 'decoder_s': t2 - t1, **stats}


def to_resolution(images, size=RESOLUTION):
    """(b, v, 3, h, w) -> (b, v, 3, *size), bilinear (no-op at the right size)."""
    if tuple(images.shape[-2:]) == tuple(size):
        return images
    b, v = images.shape[:2]
    images = F.interpolate(images.flatten(0, 1), size=size, mode='bilinear', align_corners=False)
    return images.unflatten(0, (b, v))


def stack_views(views, key):
    return torch.stack([v[key] for v in views], dim=1)


# ======================================================================================================
# Metrics (the test path of main.CSGFusion.calculate_loss with apply_mask=True, average_over_mask=False)
# ======================================================================================================

class Metrics:

    def __init__(self, device):
        self.lpips = lpips.LPIPS(pretrained=True, net='alex', verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, pred, gt, mask=None):
        """pred, gt: (b, v, 3, H, W) in [0, 1]; mask: (b, v, H, W) or None."""
        if mask is not None:
            assert mask.sum() > 0, 'There are no valid pixels in the mask!'
            pred = pred * mask[..., None, :, :]
            gt = gt * mask[..., None, :, :]
        pred, gt = pred.flatten(0, 1).clamp(0, 1), gt.flatten(0, 1)
        out = {}
        for suffix, (p, g) in (('', (pred, gt)), ('_256', (_resize256(pred), _resize256(gt)))):
            mse = ((p - g) ** 2).mean()
            out['psnr' + suffix] = (-10 * torch.log10(mse)).item()
            out['ssim' + suffix] = compute_ssim.compute_ssim(g, p, full=False).mean().item()
            out['lpips' + suffix] = self.lpips(g, p, normalize=True).mean().item()
        return out


def _resize256(x):
    return F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)


# ======================================================================================================
# Test sets
# ======================================================================================================

def scannetpp_test_set(overlap, every_n=10, root=None):
    root = root or str(ROOT / 'data' / 'scannetpp')
    return scannetpp.get_scannet_test_dataset(root, alpha=overlap, beta=overlap, resolution=list(RESOLUTION),
                                              use_every_n_sample=every_n)


def scannetpp_multiview_test_set(test_set, num_context, root=None):
    root = root or str(ROOT / 'data' / 'scannetpp')
    return scannetpp.get_scannet_multiview_test_dataset(root, str(test_set), num_context, resolution=list(RESOLUTION))


def dtu_test_set(root=None):
    root = Path(root or ROOT / 'data' / 'dtu')
    return dtu.DTUDataset(root_dir=str(root), index_json=str(root / 'test' / 'index.json'),
                          evaluation_index_json=str(ROOT / 'data' / 'dtu_index' / 'evaluation_index_dtu_nctx2.json'))


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


# ======================================================================================================
# Evaluation loop
# ======================================================================================================

def _save_png(img, path):
    import torchvision
    torchvision.utils.save_image(img.clamp(0, 1).cpu(), str(path))


def evaluate(evaluator, dataset, out_dir, use_mask=True, save_images=False, limit=None, seed=0, num_workers=4,
             time_skip=5, desc='', mask_views=None):
    """Runs ``evaluator`` over ``dataset``; writes ``samples.jsonl`` + ``summary.json`` to ``out_dir``.

    mask_views (multi-view test sets): build the evaluation mask from the first ``mask_views`` context views
    only, so that every number of input views is scored on the same pixels; the full-image metrics are then
    logged as well (``full_*``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = evaluator.device
    metrics = Metrics(device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    torch.manual_seed(seed)

    rows, skipped = [], []
    n = len(dataset) if limit is None else min(limit, len(dataset))
    with open(out_dir / 'samples.jsonl', 'w') as f:
        for i, batch in enumerate(tqdm(loader, total=n, desc=desc or evaluator.name)):
            if i >= n:
                break
            batch = to_device(batch, device)
            mask_batch = dict(batch, context=batch['context'][:mask_views]) if mask_views else batch
            mask = loss_mask.calculate_loss_mask(mask_batch) if use_mask else None
            if mask is not None and mask.sum() == 0:
                # no ground-truth pixel is visible from the context views: nothing to score, for any method
                # (main.CSGFusion.test_step skips the same samples)
                skipped.append(dict(scene=batch['scene'][0],
                                    views=[v['instance'][0] for v in batch['target'] + batch['context']]))
                continue

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            mem0 = torch.cuda.memory_allocated(device)
            out = evaluator.predict(batch)
            peak = torch.cuda.max_memory_allocated(device)

            gt = stack_views(batch['target'], 'original_img')
            row = metrics(out['target'], gt, mask)
            if mask_views:
                row.update({f'full_{k}': v for k, v in metrics(out['target'], gt).items() if k.endswith('_256')})
            ctx = metrics(out['context'], stack_views(batch['context'], 'original_img'))
            row.update({f'context_{k}': v for k, v in ctx.items() if not k.endswith('_256')})
            scene = batch['scene'][0]
            ids = [v['instance'][0] for v in batch['target'] + batch['context']]
            row.update(scene=scene, target=ids[0], context=ids[1:], encoder_s=out['encoder_s'],
                       decoder_s=out['decoder_s'], align_s=out.get('align_s', 0.0),
                       num_context=len(batch['context']),
                       **{k: out[k] for k in ('fusion_s', 'num_gaussians', 'num_gaussians_before', 'view2_dropped') if k in out}, peak_memory_mb=peak / 2 ** 20,
                       memory_increase_mb=(peak - mem0) / 2 ** 20)
            rows.append(row)
            f.write(json.dumps(row) + '\n')

            if save_images:
                d = out_dir / 'images' / scene / '_'.join([scene] + [f'{int(x):0>6}' if x.isdigit() else x for x in ids])
                d.mkdir(parents=True, exist_ok=True)
                _save_png(out['target'][0, 0], d / 'target_pred.png')
                _save_png(gt[0, 0], d / 'target_gt.png')
                if mask is not None:
                    _save_png(mask[0, 0].float(), d / 'mask.png')
                for j in range(out['context'].shape[1]):
                    _save_png(out['context'][0, j], d / f'context_{j}_pred.png')

    summary = summarize(rows, time_skip)
    summary.update(method=evaluator.name, checkpoint=str(evaluator.checkpoint),
                   parameters=evaluator.num_parameters(), masked=use_mask,
                   gt_context_poses=evaluator.gt_context_poses, target_pose_source=evaluator.target_pose_source,
                   skipped_empty_mask=len(skipped), skipped_samples=skipped)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def summarize(rows, time_skip=5):
    if not rows:
        return {'samples': 0}
    keys = [k for k, v in rows[0].items() if isinstance(v, float)]
    timed = rows[time_skip:] if len(rows) > time_skip else rows      # skip CUDA warm-up for timings
    out = {'samples': len(rows)}
    for k in keys:
        src = timed if k.endswith('_s') or k.endswith('_mb') else rows
        vals = [r[k] for r in src if math.isfinite(r[k])]
        out[k] = sum(vals) / max(len(vals), 1)
    out['total_s'] = out['encoder_s'] + out['decoder_s'] + out.get('fusion_s', 0.0)
    out['peak_memory_mb_max'] = max(r['peak_memory_mb'] for r in rows)
    return out
