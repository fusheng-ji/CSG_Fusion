"""Evaluate a CSG-Fusion checkpoint on DTU (full image, no mask).

    python scripts/eval/evaluation_dtu.py configs/evaluate_dtu.yaml [model.weights_path=<ckpt>] [key=value ...]

Writes per-scene and averaged PSNR / LPIPS / SSIM to <evaluation.save_dir>/evaluation_results/results.json.
Paths are relative to the repository root.
"""
# --- path bootstrap: run from anywhere, resolve paths relative to the repo root ---
import os as _os
import sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
# --- end bootstrap ---

import json
import os
import sys

import lightning as L
import lpips
import torch
import numpy as np

# Add MAST3R and PixelSplat to the sys.path to prevent issues during importing
sys.path.append('src/pixelsplat_src')
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')

import data.dtu as dtu
import utils.export as export
import workspace
import main

class DTUEvaluationCallback(L.Callback):
    """Lightning callback for the DTU evaluation: collects the per-sample metrics, grouped by scene."""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.scene_metrics = {}
        self.batch_count = 0

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Record the metrics returned by test_step for this batch."""
        # outputs is the return value of test_step: (loss, psnr, lpips, ssim), or None for a skipped sample
        if outputs is None:
            return

        loss, psnr, lpips_val, ssim = outputs

        # Batch information
        scene_name = batch['scene_name'][0]
        context_indices = [int(i[0]) for i in batch['context_indices']]
        target_index = batch['target_index'][0].item()

        print(f"\nProcessing batch {batch_idx + 1}...")
        print(f"  Scene: {scene_name}")
        print(f"  Context indices: {context_indices}, target index: {target_index}")

        # Metrics
        metrics = {
            'psnr': float(psnr.item()),
            'lpips': float(lpips_val.item()),
            'ssim': float(ssim.item())
        }

        # Add the metrics to their scene
        if scene_name not in self.scene_metrics:
            self.scene_metrics[scene_name] = {
                'samples': [],
                'scene_name': scene_name,
                'context_indices': context_indices,
                'target_indices': []
            }

        # Per-sample details
        sample_info = {
            'target_index': target_index,
            'metrics': metrics,
            'batch_idx': batch_idx
        }
        self.scene_metrics[scene_name]['samples'].append(sample_info)

        if target_index not in self.scene_metrics[scene_name]['target_indices']:
            self.scene_metrics[scene_name]['target_indices'].append(target_index)

        self.batch_count += 1

        print(f"  PSNR: {metrics['psnr']:.2f}, LPIPS: {metrics['lpips']:.4f}, SSIM: {metrics['ssim']:.4f}")

def run_evaluation(config):
    # Parameters from the config
    seed = getattr(config, 'seed', 42)
    devices = getattr(config, "devices", 1)

    # Load the model (model.weights_path, can be overridden on the command line)
    weights_path = config.model.weights_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = main.load_model(weights_path, device)
    model.lpips_criterion = lpips.LPIPS(net='alex').to(device)

    # Results directory
    original_save_dir = config.evaluation.save_dir
    os.makedirs(original_save_dir, exist_ok=True)

    test_dataset = dtu.DTUDataset(
        root_dir=config.data.root,
        index_json=config.data.index_json,
        evaluation_index_json=config.data.evaluation_index_path,
    )
    print(f"Dataset size: {len(test_dataset)}")

    if len(test_dataset) == 0:
        print("Error: the dataset is empty, check data.root / data.index_json.")
        return

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    # Create the output directory
    new_save_dir = os.path.join(original_save_dir, 'evaluation_results')
    os.makedirs(new_save_dir, exist_ok=True)

    L.seed_everything(seed, workers=True)

    # Evaluation callback
    eval_callback = DTUEvaluationCallback(new_save_dir)

    # Model config for the evaluation
    model.config.save_dir = new_save_dir
    model.config.loss.apply_mask = False  # no mask: compare the full rendered and ground-truth images
    model.config.loss.average_over_mask = False

    # Trainer for testing
    trainer = L.Trainer(
        accelerator="gpu",
        benchmark=True,
        callbacks=[eval_callback] + ([export.SaveAllTestBatchData(save_dir=new_save_dir)]
                                     if config.evaluation.get('save_renders', True) else []),
        default_root_dir=new_save_dir,
        devices=devices,
        log_every_n_steps=1,
        strategy=main._strategy(devices),
    )

    # Run the test
    trainer.test(model, dataloaders=data_loader_test)

    if eval_callback.batch_count == 0:
        print("Warning: no batch was processed!")
        return

    # Average metrics per scene
    scene_avg_metrics = {}
    for scene_name, metrics_list in eval_callback.scene_metrics.items():
        if len(metrics_list['samples']) > 0:
            scene_avg_metrics[scene_name] = {
                'psnr': np.mean([m['metrics']['psnr'] for m in metrics_list['samples']]),
                'lpips': np.mean([m['metrics']['lpips'] for m in metrics_list['samples']]),
                'ssim': np.mean([m['metrics']['ssim'] for m in metrics_list['samples']]),
                'num_samples': len(metrics_list['samples']),
                'scene_name': metrics_list['scene_name'],
                'context_indices': metrics_list['context_indices'],
                'target_indices': metrics_list['target_indices']
            }

    if len(scene_avg_metrics) == 0:
        print("Warning: no scene has valid metrics!")
        return

    # Average over all scenes
    all_avg_metrics = {
        'psnr': np.mean([m['psnr'] for m in scene_avg_metrics.values()]),
        'lpips': np.mean([m['lpips'] for m in scene_avg_metrics.values()]),
        'ssim': np.mean([m['ssim'] for m in scene_avg_metrics.values()]),
        'total_scenes': len(scene_avg_metrics),
        'total_samples': sum([m['num_samples'] for m in scene_avg_metrics.values()])
    }

    # Save the results
    results = {
        'scene_metrics': scene_avg_metrics,
        'average_metrics': all_avg_metrics,
        'config': {
            'model_weights': config.model.weights_path,
            'data_root': config.data.root,
            'batch_size': int(config.data.batch_size),
            'evaluation_index': config.data.evaluation_index_path
        }
    }

    save_path = os.path.join(new_save_dir, 'results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to: {save_path}")

    # Summary
    print(f"Processed {eval_callback.batch_count} batches, {len(scene_avg_metrics)} scenes")

    print("\nMetrics per scene:")
    for scene_name, metrics in scene_avg_metrics.items():
        print(f"\n{scene_name} ({metrics['scene_name']}):")
        print(f"  Samples: {metrics['num_samples']}")
        print(f"  Context indices: {metrics['context_indices']}")
        print(f"  Target indices: {metrics['target_indices']}")
        print(f"  PSNR: {metrics['psnr']:.3f}")
        print(f"  LPIPS: {metrics['lpips']:.4f}")
        print(f"  SSIM: {metrics['ssim']:.4f}")

    print("\nAverage over all scenes:")
    print(f"  PSNR: {all_avg_metrics['psnr']:.3f}")
    print(f"  LPIPS: {all_avg_metrics['lpips']:.4f}")
    print(f"  SSIM: {all_avg_metrics['ssim']:.4f}")
    print(f"  Scenes: {all_avg_metrics['total_scenes']}")
    print(f"  Samples: {all_avg_metrics['total_samples']}")

    # Scenes ranked from best to worst
    if len(scene_avg_metrics) > 1:
        print("\nScene ranking (by PSNR):")
        sorted_scenes = sorted(scene_avg_metrics.items(), key=lambda x: x[1]['psnr'], reverse=True)
        for i, (scene_name, metrics) in enumerate(sorted_scenes):
            print(f"  {i+1:2d}. {scene_name}: PSNR={metrics['psnr']:.2f}, LPIPS={metrics['lpips']:.3f}, SSIM={metrics['ssim']:.3f}")

if __name__ == "__main__":
    # Load the config (key=value arguments override it, e.g. model.weights_path=<ckpt>)
    config = workspace.load_config(sys.argv[1], sys.argv[2:])
    if os.getenv("LOCAL_RANK", '0') == '0':
        config = workspace.create_workspace(config)

    # Run the evaluation
    run_evaluation(config)
