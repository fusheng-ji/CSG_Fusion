"""Evaluate a baseline on the CSG-Fusion ScanNet++ (masked, four overlap levels) or DTU test set.

    python -m scripts.eval.baselines --method pixelsplat
    python -m scripts.eval.baselines --method splatt3r --dataset dtu

Multi-view ScanNet++ (scripts/eval/make_multiview_testset.py), e.g. CSG-Fusion on 2, 4 and 8 views:

    python -m scripts.eval.baselines --method csg_fusion \
        --dataset scannetpp_multiview --num-context 2 4 8 --mv-mode pairwise --mv-filter all

Results: <out>/<method>/<dataset>[/overlap_<a> | /views_<n>]/{summary.json, samples.jsonl, images/}, plus
<out>/<method>/<dataset>.json
"""
import argparse
import json
from pathlib import Path

import torch

from . import CHECKPOINTS, METHODS, get_evaluator
from .common import ROOT, dtu_test_set, evaluate, scannetpp_multiview_test_set, scannetpp_test_set


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', required=True, choices=sorted(METHODS))
    p.add_argument('--checkpoint', default=None, help='default: the path in scripts/eval/baselines/__init__.py (see README)')
    p.add_argument('--dataset', default='scannetpp', choices=['scannetpp', 'scannetpp_multiview', 'dtu'])
    p.add_argument('--overlaps', nargs='+', type=float, default=[0.9, 0.7, 0.5, 0.3], help='ScanNet++ only')
    p.add_argument('--every-n', type=int, default=10, help='use every n-th test sample (CSG-Fusion uses 10)')
    p.add_argument('--data-root', default=None, help='default: data/scannetpp or data/dtu')
    p.add_argument('--out', default=str(ROOT / 'results' / 'baselines'))
    p.add_argument('--limit', type=int, default=None, help='only the first N samples per test set (smoke test)')
    p.add_argument('--save-images', action='store_true', help='save target / context renders, ground truth, mask')
    p.add_argument('--no-mask', action='store_true', help='ScanNet++: evaluate the full image instead of the mask')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--fusion', default='none', choices=['none', 'depth', 'merge', 'random'],
                   help='training-free cross-view fusion (src/fusion_plugin.py): drop view-2 Gaussians that '
                        'duplicate view-1 geometry (depth) or the same fraction at random (control)')
    p.add_argument('--fusion-tau', type=float, default=0.01, help='depth: relative depth tolerance')
    p.add_argument('--fusion-ratio', type=float, default=None, help='random: fraction of view-2 Gaussians')
    mv = p.add_argument_group('multi-view (--dataset scannetpp_multiview)')
    mv.add_argument('--test-set', default=str(ROOT / 'data' / 'scannetpp' / 'test_set_multiview_0.5.json'))
    mv.add_argument('--num-context', nargs='+', type=int, default=[2, 4, 8], help='numbers of input views')
    mv.add_argument('--mask-views', type=int, default=2,
                    help='evaluation mask from the first n context views (the same pixels for every N)')
    mv.add_argument('--mv-mode', default=None, choices=['pairwise', 'anchored', 'joint'],
                    help='csg_fusion: multi-view inference mode (src/multiview.py; default pairwise for N > 2)')
    mv.add_argument('--mv-filter', default=None, choices=['none', 'ref', 'all'], help='csg_fusion (default all)')
    mv.add_argument('--ref-fusion', default=None, choices=['concat', 'best'], help='csg_fusion, joint mode (default concat)')
    p.add_argument('overrides', nargs='*', help='extra Hydra overrides for the baselines that use Hydra')
    args = p.parse_args()
    args.checkpoint = args.checkpoint or ROOT / 'checkpoint' / CHECKPOINTS[args.method]

    evaluator = get_evaluator(args.method)(args.checkpoint, device=args.device, overrides=args.overrides)
    poses = 'GT context poses' if evaluator.gt_context_poses else 'no context poses (pose-free)'
    print(f'{args.method}: {evaluator.num_parameters():,} parameters, {poses}')
    evaluator.fusion_mode, evaluator.fusion_tau, evaluator.fusion_ratio = args.fusion, args.fusion_tau, args.fusion_ratio
    if args.fusion == 'random' and args.fusion_ratio is None:
        p.error('--fusion random needs --fusion-ratio')
    tag = '' if args.fusion == 'none' else (f'_fusion-random-{args.fusion_ratio:g}' if args.fusion == 'random'
                                            else f'_fusion-{args.fusion}-{args.fusion_tau:g}')
    if (args.mv_mode, args.mv_filter, args.ref_fusion) != (None, None, None):
        if args.method != 'csg_fusion':
            p.error('--mv-mode / --mv-filter / --ref-fusion are for --method csg_fusion')
        mode, flt, ref = args.mv_mode or 'pairwise', args.mv_filter or 'all', args.ref_fusion or 'concat'
        evaluator.mv_mode, evaluator.mv_filter, evaluator.ref_fusion = mode, flt, ref
        tag += f'_mv-{mode}-{flt}' + (f'-{ref}' if mode == 'joint' else '')
    out = Path(args.out) / (args.method + tag)
    common = dict(save_images=args.save_images, limit=args.limit, seed=args.seed, num_workers=args.num_workers)

    results = {}
    if args.dataset == 'scannetpp':
        for a in args.overlaps:
            ds = scannetpp_test_set(a, every_n=args.every_n, root=args.data_root)
            results[f'overlap_{a}'] = evaluate(evaluator, ds, out / 'scannetpp' / f'overlap_{a}',
                                               use_mask=not args.no_mask, desc=f'{args.method} overlap {a}', **common)
    elif args.dataset == 'scannetpp_multiview':
        for n in args.num_context:
            ds = scannetpp_multiview_test_set(args.test_set, n, root=args.data_root)
            results[f'views_{n}'] = evaluate(evaluator, ds, out / args.dataset / f'views_{n}', use_mask=not args.no_mask,
                                             mask_views=args.mask_views, desc=f'{args.method} {n} views', **common)
    else:
        # DTU is evaluated on the full, un-masked image (as CSG-Fusion's scripts/eval/evaluation_dtu.py)
        results['dtu'] = evaluate(evaluator, dtu_test_set(args.data_root), out / 'dtu', use_mask=False,
                                  desc=f'{args.method} dtu', **common)

    results['gt_context_poses'] = evaluator.gt_context_poses
    if args.fusion != 'none':
        results['fusion'] = dict(mode=args.fusion, tau=args.fusion_tau, ratio=args.fusion_ratio)
    with open(out / f'{args.dataset}.json', 'w') as f:
        json.dump(results, f, indent=2)
    results.pop('gt_context_poses')
    results.pop('fusion', None)
    keys = ('psnr_256', 'ssim_256', 'lpips_256', 'total_s', 'peak_memory_mb')
    print(f'{"":>12}' + ''.join(f'{k:>16}' for k in keys))
    for name, r in results.items():
        print(f'{name:>12}' + ''.join(f'{r.get(k, float("nan")):>16.4f}' for k in keys))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
