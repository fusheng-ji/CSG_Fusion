"""Latency, GPU / host memory, compute and Gaussian count of CSG-Fusion and the baselines on two-view pairs.

    python -m scripts.eval.bench_efficiency pairs                                  # once: the pairs, as files
    python -m scripts.eval.bench_efficiency run --method mvsplat --pass 0          # one process per method and pass
    python -m scripts.eval.bench_efficiency all                                    # every method, three passes
    python -m scripts.eval.summarize_efficiency                                    # -> results/efficiency/summary.json

The same 100 ScanNet++ test pairs for every method (25 per overlap), batch 1, each method at its own resolution
and with its own code. Use an idle GPU. Every process writes results/efficiency/<method>_pass<k>.json, and
``all`` skips the files that exist.

* Latency: CUDA-synchronised wall time after warm-up, without emptying the CUDA cache between samples.
  ``reconstruct``: the two images -> Gaussians. ``encode``: what the evaluation runs, i.e. ``reconstruct`` plus the
  prediction of the target camera where the method does that; the difference is reported as ``camera``.
  ``render``: the target view. Input preparation is not timed. YoNoSplat's test-time alignment is timed separately.
* GPU memory: peak allocated memory of ``reconstruct`` and ``render``, including the weights.
* Host memory: peak resident set size of the process during the timed loop.
* Compute: multiply-adds of ``reconstruct`` (matrix products, convolutions and attention), counted with
  ``torch.utils.flop_counter``. CSG-Fusion's CUDA graphs are off for this, as the counter cannot see a replay.

The summary is relative to CSG-Fusion for times; the raw files keep seconds.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from .baselines import CHECKPOINTS, get_evaluator
from .baselines.common import ROOT, scannetpp_test_set, to_device

OUT = ROOT / 'results' / 'efficiency'
OVERLAPS, PER_OVERLAP, WARMUP, FLOP_PAIRS, PASSES = (0.9, 0.7, 0.5, 0.3), 25, 10, 8, 3
RUNS = list(CHECKPOINTS)


def make_pairs(args):
    """Evenly spaced pairs of every test set, interleaved by overlap, so that every warm-up and subset is mixed."""
    (OUT / 'pairs').mkdir(parents=True, exist_ok=True)
    for j, overlap in enumerate(OVERLAPS):
        ds = scannetpp_test_set(overlap)
        for k in range(PER_OVERLAP):
            batch = torch.utils.data.default_collate([ds[round(k * (len(ds) - 1) / (PER_OVERLAP - 1))]])
            torch.save(batch, OUT / 'pairs' / f'{k * len(OVERLAPS) + j:03d}.pt')


def host_memory():
    """Resident set size of this process, now and peak, in MiB."""
    status = dict(line.split(':', 1) for line in open('/proc/self/status') if line.startswith(('VmRSS', 'VmHWM')))
    return {k: int(v.split()[0]) / 1024 for k, v in status.items()}


def reset_host_peak():
    with open('/proc/self/clear_refs', 'w') as f:
        f.write('5')


def gpu_load():
    """Other processes' view of the GPU: utilisation and memory, to spot a shared device."""
    r = subprocess.run(['nvidia-smi', '--query-gpu=uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True)
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    return next((line.split(', ')[1:] for line in r.stdout.splitlines() if uuid in line), None)


def count_flops(evaluator, inputs):
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as counter:
        evaluator.reconstruct(inputs)
    return {str(op): n for op, n in counter.get_flop_counts()['Global'].items()}


@torch.no_grad()
def run(args):
    torch.backends.cudnn.benchmark = True                           # as the evaluation
    method = args.method
    load = [gpu_load()]
    evaluator = get_evaluator(method)(ROOT / 'checkpoint' / CHECKPOINTS[method])
    device = evaluator.device
    align = method == 'yonosplat' and evaluator.test_cfg.align_pose
    if align:
        evaluator.test_cfg.align_pose = False
    host_load = host_memory()['VmHWM']
    torch.cuda.empty_cache()
    weights = torch.cuda.memory_allocated(device)
    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((OUT / 'pairs').glob('*.pt'))[:args.limit]
    load_pair = lambda f: to_device(torch.load(f, weights_only=False), device)

    def now():
        torch.cuda.synchronize(device)
        return time.perf_counter()

    def timed(inputs):
        t0 = now()
        out = evaluator.reconstruct(inputs)
        t1 = now()
        count = evaluator.count_gaussians(out)
        camera = 0.0
        if evaluator.target_pose_source.startswith('predicted'):    # reconstruct does not give the target camera
            del out
            t2 = now()
            out = evaluator.encode(inputs)
            camera = now() - t2
        t3 = now()
        image = evaluator.render(out, inputs, 'target')
        return {'reconstruct_s': t1 - t0, 'encode_s': camera or t1 - t0, 'render_s': now() - t3, 'gaussians': count,
                'resolution': list(image.shape[-2:])}

    rows, warm = [], (files * WARMUP)[:WARMUP]
    with evaluator.sandbox:
        for f in warm:
            timed(evaluator.prepare(load_pair(f)))
        reset_host_peak()
        for f in files:
            rows.append(timed(evaluator.prepare(load_pair(f))))
        host_peak = host_memory()['VmHWM']

        for f, row in zip(files, rows):                             # memory, in a loop of its own
            inputs = evaluator.prepare(load_pair(f))
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            out = evaluator.reconstruct(inputs)
            row['reconstruct_peak'], row['reserved'] = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
            if evaluator.target_pose_source.startswith('predicted'):
                del out
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                out = evaluator.encode(inputs)
                row['encode_peak'] = torch.cuda.max_memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            evaluator.render(out, inputs, 'target')
            row['render_peak'] = torch.cuda.max_memory_allocated(device)
            del out, inputs

        flops, align_s = [], []
        if args.index == 0:
            if method == 'csg_fusion':
                for g in evaluator.model.backbone_graphs:
                    g.enabled = False
            for module in evaluator.modules():                      # the counter's module hooks need plain tensors
                module.requires_grad_(False)
            for f in files[:FLOP_PAIRS]:
                flops.append(count_flops(evaluator, evaluator.prepare(load_pair(f))))
            if align:
                evaluator.test_cfg.align_pose = True
                for f in files[:FLOP_PAIRS]:
                    inputs = evaluator.prepare(load_pair(f))
                    evaluator.render(evaluator.encode(inputs), inputs, 'target')
                    align_s.append(evaluator.last_align_s)
    load.append(gpu_load())
    (out_dir / f'{method}_pass{args.index}.json').write_text(json.dumps(dict(
        method=method, parameters=evaluator.num_parameters(), weights=weights, host_load_mb=host_load,
        host_peak_mb=host_peak, flops=flops, align_s=align_s, gpu_load=load, device=torch.cuda.get_device_name(0),
        target_pose_source=evaluator.target_pose_source, rows=rows)))
    med = lambda k: 1e3 * statistics.median(r[k] for r in rows)
    print(f'{method:22s} pass {args.index}: reconstruct {med("reconstruct_s"):7.1f}  encode {med("encode_s"):7.1f}  '
          f'render {med("render_s"):6.1f} ms  Gaussians {statistics.mean(r["gaussians"] for r in rows):9.0f}  '
          f'peak {max(max(r["reconstruct_peak"], r["render_peak"]) for r in rows) / 2**30:5.2f} GiB  host {host_peak / 1024:5.2f} GiB  load {load}',
          flush=True)


def run_all(args):
    for k in range(PASSES):
        for method in RUNS[4 * k:] + RUNS[:4 * k]:                  # rotated, so that a drift hits all methods alike
            if not (OUT / f'{method}_pass{k}.json').exists():
                r = subprocess.run([sys.executable, '-m', 'scripts.eval.bench_efficiency', 'run', '--method', method,
                                    '--pass', str(k)], cwd=ROOT, check=False)
                if r.returncode:
                    print(f'{method} pass {k} failed (checkpoint or submodule missing?)', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    sub.add_parser('pairs').set_defaults(func=make_pairs)
    r = sub.add_parser('run')
    r.add_argument('--method', required=True, choices=RUNS)
    r.add_argument('--pass', dest='index', type=int, default=0)
    r.add_argument('--out', default=None, help='default: results/efficiency')
    r.add_argument('--limit', type=int, default=None, help='only the first pairs (smoke test)')
    r.set_defaults(func=run)
    sub.add_parser('all').set_defaults(func=run_all)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
