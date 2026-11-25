"""Evaluate a CSG-Fusion checkpoint on the masked ScanNet++ test set (four overlap levels).

    python scripts/eval/evaluation_scannetpp.py --checkpoint checkpoint/csg_fusion.ckpt

For every overlap level the test samples are run through `trainer.test` (CSGFusion.test_step: PSNR / SSIM /
LPIPS on the loss mask). The script also records the parameter count, the wall time of the whole test loop and
the peak GPU memory, and writes everything to <out>/results.json (Lightning's benchmark.json per overlap goes to
<out>/overlap_<o>/). Runs on one GPU so that time and memory are comparable between checkpoints.

Paths are relative to the repository root. The baseline harness (python -m scripts.eval.baselines --method
csg_fusion ...) gives the same metrics per sample with the code shared by all baselines.
"""
# --- path bootstrap: run from anywhere, resolve paths relative to the repo root ---
import os as _os
import sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
# --- end bootstrap ---

import argparse
import json
import os
import time

import lightning as L
import lpips
import torch

import main  # first: puts MASt3R / DUSt3R / PixelSplat on sys.path
import data.scannetpp.scannetpp as scannetpp
import workspace

DEFAULT_CONFIG = 'configs/base.yaml'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=DEFAULT_CONFIG,
                   help='config for data.root, data.resolution, data.batch_size, data.num_workers and seed '
                        f'(default: {DEFAULT_CONFIG}); the model itself is rebuilt from the checkpoint')
    p.add_argument('--checkpoint', required=True, help='CSG-Fusion checkpoint (.ckpt)')
    p.add_argument('--overlaps', nargs='+', type=float, default=[0.9, 0.7, 0.5, 0.3],
                   help='overlap levels to evaluate (data/scannetpp/test_set_<o>_<o>.json)')
    p.add_argument('--every-n', type=int, default=10, help='use every n-th test sample (default: 10)')
    p.add_argument('--out', default=None,
                   help='output directory (default: results/eval_scannetpp/<checkpoint run name>)')
    p.add_argument('--average-over-mask', action='store_true',
                   help='average the metrics over the masked pixels only (default: mask both images, average '
                        'over the full image)')
    p.add_argument('overrides', nargs='*', help='extra key=value overrides of --config (e.g. data.root=...)')
    return p.parse_args()


def run_name(checkpoint):
    """`results/<run>/checkpoints/x.ckpt` -> `<run>` (may contain '/'); otherwise the checkpoint's file stem."""
    path = os.path.normpath(checkpoint)
    parent = os.path.dirname(path)
    if os.path.basename(parent) != 'checkpoints':
        return os.path.splitext(os.path.basename(path))[0]
    run_dir = os.path.dirname(parent)
    parts = run_dir.split(os.sep)
    if 'results' in parts[:-1]:  # the part after the last 'results/'
        return os.path.join(*parts[len(parts) - parts[::-1].index('results'):])
    return os.path.basename(run_dir) or 'run'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(args, config):
    out_dir = args.out or os.path.join('results', 'eval_scannetpp', run_name(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda')

    model = main.load_model(args.checkpoint, device)
    num_params = count_parameters(model)
    print(f'Trainable parameters: {num_params:,}')

    apply_mask, average_over_mask = True, bool(args.average_over_mask)
    model.config.loss.apply_mask = apply_mask
    model.config.loss.average_over_mask = average_over_mask
    model.lpips_criterion = lpips.LPIPS(net='alex', spatial=average_over_mask).to(device)

    results = {
        'checkpoint': args.checkpoint,
        'config': args.config,
        'data_root': config.data.root,
        'apply_mask': apply_mask,
        'average_over_mask': average_over_mask,
        'every_n': args.every_n,
        'model_parameters': num_params,
        'overlaps': {},
    }
    save_path = os.path.join(out_dir, 'results.json')

    for overlap in args.overlaps:
        test_dataset = scannetpp.get_scannet_test_dataset(
            config.data.root,
            alpha=overlap,
            beta=overlap,
            resolution=config.data.resolution,
            use_every_n_sample=args.every_n,
        )
        data_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=1,              # PSNR per image: a larger batch would pool the squared error over images
            **main.dataloader_kwargs(config),
        )

        overlap_dir = os.path.join(out_dir, f'overlap_{overlap}')
        os.makedirs(overlap_dir, exist_ok=True)
        model.config.save_dir = overlap_dir  # CSGFusion.on_test_end writes benchmark.json here

        L.seed_everything(config.seed, workers=True)
        trainer = L.Trainer(
            accelerator='gpu',
            devices=1,
            benchmark=True,
            logger=False,
            enable_checkpointing=False,
            default_root_dir=overlap_dir,
        )

        print(f'Evaluating overlap {overlap} ({len(test_dataset)} samples)')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated() / 2 ** 20
        start_time = time.time()

        res = trainer.test(model, dataloaders=data_loader)

        torch.cuda.synchronize()
        total_time = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated() / 2 ** 20
        n = len(test_dataset)

        results['overlaps'][f'overlap_{overlap}'] = {
            'test_results': res,
            'dataset_size': n,
            'total_time_s': total_time,
            'time_per_sample_s': total_time / n if n else None,
            'gpu_memory_start_mb': start_mem,
            'peak_gpu_memory_mb': peak_mem,
            'peak_gpu_memory_increase_mb': peak_mem - start_mem,
        }
        print(f'  wall time {total_time:.2f} s ({total_time / max(n, 1):.4f} s / sample), '
              f'peak GPU memory {peak_mem:.1f} MB (+{peak_mem - start_mem:.1f} MB)')

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved to {save_path}')


if __name__ == '__main__':
    args = parse_args()
    config = workspace.load_config(args.config, args.overrides)
    evaluate(args, config)
