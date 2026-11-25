"""Baseline evaluation: unmodified upstream repos (third_party/baselines/) under CSG-Fusion's test protocol.

    python -m scripts.eval.baselines --method noposplat
"""

METHODS = {
    'pixelsplat': ('re10k_family', 'PixelSplatEvaluator'),
    'mvsplat': ('re10k_family', 'MVSplatEvaluator'),
    'noposplat': ('re10k_family', 'NoPoSplatEvaluator'),
    'splatt3r': ('splatt3r', 'Splatt3REvaluator'),
    'depthsplat': ('depthsplat', 'DepthSplatEvaluator'),
    'anysplat': ('anysplat', 'AnySplatEvaluator'),
    'yonosplat': ('yonosplat', 'YoNoSplatEvaluator'),
    'spfsplat': ('spfsplat', 'SPFSplatEvaluator'),
    'spfsplatv2_l': ('spfsplat', 'SPFSplatV2LEvaluator'),
    'csg_fusion': ('csg_fusion', 'CSGFusionEvaluator'),     # ours, under the same protocol
}

CHECKPOINTS = {          # default checkpoints, under checkpoint/ (see README "Checkpoints")
    'pixelsplat': 'baselines/pixelsplat_re10k.ckpt',
    'mvsplat': 'baselines/mvsplat_re10k.ckpt',
    'noposplat': 'baselines/noposplat_re10k.ckpt',
    'splatt3r': 'epoch=19-step=1200.ckpt',
    'depthsplat': 'baselines/depthsplat_large_re10k.pth',
    'anysplat': 'baselines/anysplat',
    'yonosplat': 'baselines/yonosplat_re10k.ckpt',
    'spfsplat': 'baselines/spfsplat_re10k.ckpt',
    'spfsplatv2_l': 'baselines/spfsplatv2l_re10k.ckpt',
    'csg_fusion': 'csg_fusion.ckpt',
}


def get_evaluator(name):
    import importlib
    module, cls = METHODS[name]
    return getattr(importlib.import_module(f'{__name__}.{module}'), cls)
