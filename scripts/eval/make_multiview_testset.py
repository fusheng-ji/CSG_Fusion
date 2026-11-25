"""Build a multi-view ScanNet++ test set from a two-view one.

    python -m scripts.eval.make_multiview_testset --overlap 0.5 --every-n 50 --num-context 8

Each sample [scene, c1, c2, target] of data/scannetpp/test_set_<overlap>_<overlap>.json (every n-th) keeps its
two context views and its target and gets up to ``num_context`` context views in total. The extra views are
drawn at random (seeded per sample) from the frames of the same scene that, by the Splatt3R coverage
matrices (data/scannetpp/coverage/<scene>.json, used symmetrically as max(C[i][j], C[j][i])):

    overlap a chosen context view by more than --min-overlap   (the views form one connected capture)
    are not near-copies of a chosen view or of the target      (coverage below --max-overlap)

The context lists are nested: the first N entries are the N-view input, and N = 2 is the original two-view
sample. Samples whose scene has too few such frames are dropped. Output: [scene, [c1, ..., cN], target].
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def symmetric(cov, i, j):
    return max(cov[i][j], cov[j][i])


def extend(cov, context, target, n, rng, min_overlap, max_overlap):
    chosen = list(context)
    frames = range(len(cov))
    while len(chosen) < n:
        candidates = [f for f in frames if f not in chosen and f != target
                      and symmetric(cov, f, target) < max_overlap
                      and all(symmetric(cov, f, c) < max_overlap for c in chosen)
                      and any(symmetric(cov, f, c) > min_overlap for c in chosen)]
        if not candidates:
            return None
        chosen.append(rng.choice(candidates))
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--overlap', default='0.5', help='two-view test set to start from')
    p.add_argument('--every-n', type=int, default=50)
    p.add_argument('--num-context', type=int, default=8)
    p.add_argument('--min-overlap', type=float, default=0.3)
    p.add_argument('--max-overlap', type=float, default=0.9)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None, help='default: data/scannetpp/test_set_multiview_<overlap>.json')
    args = p.parse_args()

    data = ROOT / 'data' / 'scannetpp'
    samples = json.load(open(data / f'test_set_{args.overlap}_{args.overlap}.json'))[::args.every_n]
    coverage = {}
    out, dropped = [], 0
    for idx, (scene, c1, c2, target) in enumerate(samples):
        if scene not in coverage:
            coverage[scene] = json.load(open(data / 'coverage' / f'{scene}.json'))[scene]
        rng = random.Random(f'{args.seed}-{idx}-{scene}-{c1}-{c2}-{target}')
        context = extend(coverage[scene], [c1, c2], target, args.num_context, rng, args.min_overlap, args.max_overlap)
        if context is None:
            dropped += 1
            continue
        out.append([scene, context, target])

    path = Path(args.out) if args.out else data / f'test_set_multiview_{args.overlap}.json'
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f'{len(out)} samples ({dropped} of {len(samples)} dropped: too few frames), '
          f'{len({s[0] for s in out})} scenes -> {path}')


if __name__ == '__main__':
    main()
