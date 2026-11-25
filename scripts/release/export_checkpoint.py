"""Turn a training checkpoint into the release checkpoint: model weights + config only.

    python scripts/release/export_checkpoint.py results/<run>/checkpoints/last.ckpt checkpoint/csg_fusion.ckpt

Works for both backbones (MASt3R and VGGT). Lightning checkpoints also hold the optimiser state, which inference does not need. The output keeps exactly
what `main.load_model` reads (`state_dict` and `hyper_parameters.config`), without paths and settings of the training run.
"""
import argparse
import os
import sys

import omegaconf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import main  # noqa: E402


def export():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('src')
    parser.add_argument('dst')
    args = parser.parse_args()
    ckpt = torch.load(args.src, map_location='cpu', weights_only=False)
    config = ckpt['hyper_parameters']['config']
    if isinstance(config, omegaconf.DictConfig):
        config = omegaconf.OmegaConf.to_container(config)  # plain dict: no pickled omegaconf objects
    # Bookkeeping of the training run that means nothing elsewhere; the model keys are left untouched, so the
    # released checkpoint rebuilds exactly the same network.
    config['name'] = os.path.splitext(os.path.basename(args.dst))[0]
    config['save_dir'] = f'./results/{config["name"]}/'
    config.pop('include', None)  # already merged into the stored config
    config.setdefault('loggers', {})['use_wandb'] = False
    config.setdefault('data', {})['root'] = './data/scannetpp'
    for key in ('train', 'trainer', 'eval'):
        config.pop(key, None)
    if 'model' in config:
        config['model'] = {k: v for k, v in config['model'].items() if k.startswith('vggt_') or f'model.{k}' in main.DEFAULTS}
        if 'vggt_pretrained' in config['model']:
            config['model']['vggt_pretrained'] = 'facebook/VGGT-1B'
    torch.save({'state_dict': ckpt['state_dict'], 'hyper_parameters': {'config': config}}, args.dst)
    n = sum(v.numel() for v in ckpt['state_dict'].values())
    print(f'{args.dst}: {len(ckpt["state_dict"])} tensors, {n / 1e6:.0f}M values')


if __name__ == '__main__':
    export()
