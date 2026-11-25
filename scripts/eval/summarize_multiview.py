"""Table of multi-view results written by ``python -m scripts.eval.baselines --dataset scannetpp_multiview``.

    python -m scripts.eval.summarize_multiview results/multiview/mast3r results/multiview/vggt

For every run folder (<root>/<method>[_mv-<mode>-<filter>[-<ref fusion>]]) and every number of views: PSNR / SSIM /
LPIPS at 256 px on the evaluation mask of the first two context views (the same pixels for every N), the
full-image PSNR, the number of Gaussians, the share of Gaussians the filter removes (of all N x H x W pixel
Gaussians, and of those of views 2..N; the reference view keeps all of its own) and the seconds per scene.
"""
import argparse
import json
from pathlib import Path


def rows(root):
    for run in sorted(Path(root).iterdir()):
        f = run / 'scannetpp_multiview.json'
        if not f.exists():
            continue
        res = json.load(open(f))
        for key in sorted((k for k in res if k.startswith('views_')), key=lambda k: int(k.split('_')[1])):
            r = res[key]
            yield (f'{Path(root).name}/{run.name}', int(key.split('_')[1]), r['samples'], r['psnr_256'], r['ssim_256'],
                   r['lpips_256'], r.get('full_psnr_256', float('nan')), r.get('num_gaussians', float('nan')),
                   100 * (1 - r['num_gaussians'] / r['num_gaussians_before']) if 'num_gaussians_before' in r else float('nan'),
                   100 * r.get('view2_dropped', float('nan')), r['total_s'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('roots', nargs='+')
    p.add_argument('--markdown', action='store_true')
    args = p.parse_args()
    head = ('run', 'N', 'samples', 'PSNR', 'SSIM', 'LPIPS', 'full PSNR', 'Gaussians (k)', 'removed %',
            'views 2..N removed %', 's / scene')
    out = [r for root in args.roots for r in rows(root)]
    if args.markdown:
        print('| ' + ' | '.join(head) + ' |')
        print('|' + '---|' * len(head))
        for r in out:
            print(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]:.2f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.2f} | {r[7] / 1e3:.0f} '
                  f'| {r[8]:.1f} | {r[9]:.1f} | {r[10]:.2f} |')
        return
    print(f'{head[0]:<48}' + ''.join(f'{h:>22}' for h in head[1:]))
    for r in out:
        print(f'{r[0]:<48}{r[1]:>22}{r[2]:>22}{r[3]:>22.2f}{r[4]:>22.3f}{r[5]:>22.3f}{r[6]:>22.2f}{r[7] / 1e3:>22.0f}'
              f'{r[8]:>22.1f}{r[9]:>22.1f}{r[10]:>22.2f}')


if __name__ == '__main__':
    main()
