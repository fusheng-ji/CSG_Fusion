"""Summary of scripts/eval/bench_efficiency.py: a JSON file and the README table (methods without results are left out).

    python -m scripts.eval.summarize_efficiency        # results/efficiency/*.json -> summary.json, table on stdout

Times are relative to CSG-Fusion (reconstruct + render = 1); medians over all pairs and passes.
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'results' / 'efficiency'
GROUPS = {'Ground-truth context poses': {'pixelsplat': 'pixelSplat', 'mvsplat': 'MVSplat', 'depthsplat': 'DepthSplat'},
          'No context poses': {'noposplat': 'NoPoSplat', 'splatt3r': 'Splatt3R', 'csg_fusion': 'CSG-Fusion'},
          'No context poses, target camera estimated': {'spfsplat': 'SPFSplat', 'spfsplatv2_l': 'SPFSplatV2-L',
                                                        'yonosplat': 'YoNoSplat', 'anysplat': 'AnySplat'}}
# how the Gaussians come about: one set per pixel (default), per pixel and then filtered, or merged in voxels
FILTERED, VOXEL = {'csg_fusion', 'yonosplat'}, {'anysplat'}
KINDS = {'pixel': 'pixel-wise', 'filtered': 'filtered', 'voxel': 'voxel-merged'}
NAMES = {m: name for group in GROUPS.values() for m, name in group.items()}


def load(method):
    runs = [json.load(open(f)) for f in sorted(RAW.glob(f'{method}_pass*.json'))]
    rows = [r for run in runs for r in run['rows']]
    med = lambda k: 1e3 * st.median(r[k] for r in rows)
    peak = lambda run: max(max(r['reconstruct_peak'], r['render_peak']) for r in run['rows'])
    flops = [sum(f.values()) for run in runs for f in run['flops']]
    align = [a for run in runs for a in run['align_s']]
    return dict(passes=len(runs), pairs=len(runs[0]['rows']), reconstruct=med('reconstruct_s'),
                camera=max(med('encode_s') - med('reconstruct_s'), 0.0), render=med('render_s'),
                quartiles=st.quantiles([1e3 * (r['reconstruct_s'] + r['render_s']) for r in rows], n=4)[::2],
                align=1e3 * st.median(align) if align else 0.0,
                per_pass=[1e3 * st.median(r['reconstruct_s'] + r['render_s'] for r in run['rows']) for run in runs],
                gpu=max(map(peak, runs)) / 2**30, weights=runs[0]['weights'] / 2**30,
                host=max(run['host_peak_mb'] for run in runs) / 1024, gmacs=st.mean(flops) / 2e9,
                gaussians=st.mean(r['gaussians'] for r in rows), parameters=runs[0]['parameters'] / 1e6,
                resolution=rows[0]['resolution'][0])


def summarize():
    data = {m: load(m) for m in NAMES if list(RAW.glob(f'{m}_pass*.json'))}
    if 'csg_fusion' not in data:
        raise SystemExit('no csg_fusion results: times are relative to it')
    unit = data['csg_fusion']['reconstruct'] + data['csg_fusion']['render']
    for d in data.values():
        d['total'] = d['reconstruct'] + d['camera'] + d['render']
        for k in ('reconstruct', 'camera', 'render', 'total', 'align'):
            d[k] = round(d[k] / unit, 3)
        d['quartiles'] = [round(q / unit, 3) for q in d['quartiles']]
        d['spread'] = round(max(d['per_pass']) / min(d['per_pass']) - 1, 3)     # between the passes
        del d['per_pass']
        for k, n in (('gpu', 2), ('weights', 2), ('host', 2), ('gmacs', 0), ('gaussians', 0), ('parameters', 0)):
            d[k] = round(d[k], n)
    return data


def kind(method):
    return 'filtered' if method in FILTERED else 'voxel' if method in VOXEL else 'pixel'


def cells(method, d):
    """One table row: the strings shown in the README and on the project page."""
    camera = f'{d["camera"]:.2f}×' if d['camera'] else (f'({d["align"]:.0f}×)' if d['align'] else '–')
    return [NAMES[method], f'{d["reconstruct"]:.2f}×', camera, f'{d["render"]:.2f}×', f'{d["total"]:.2f}×',
            f'{d["gpu"]:.1f}', f'{d["weights"]:.1f}' if d["weights"] >= 0.05 else '<0.1', f'{d["host"]:.1f}', f'{d["gmacs"]:,.0f}',
            KINDS[kind(method)] + (', 3 each' if method == 'pixelsplat' else ''),
            f'{d["gaussians"] / 1e3:.0f}k', f'{d["parameters"]:.0f}M', f'{d["resolution"]}']


HEAD = ['Method', 'Reconstruct', 'Target camera', 'Render', 'Total', 'Peak GPU memory (GiB)', 'of which weights',
        'Peak host memory (GiB)', 'GMACs', 'Gaussian type', 'Gaussians', 'Parameters', 'Pixels per side']


def markdown(data):
    lines = ['| ' + ' | '.join(HEAD) + ' |', '|---|' + '---:|' * (len(HEAD) - 1)]
    for m, d in data.items():
        row = cells(m, d)
        lines.append('| ' + ' | '.join(f'**{c}**' if m == 'csg_fusion' else c for c in row) + ' |')
    return '\n'.join(lines)


if __name__ == '__main__':
    summary = summarize()
    (RAW / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(markdown(summary))
    print('\nspread between the passes: ' + ', '.join(f'{m} {d["spread"]:.1%}' for m, d in summary.items()))
