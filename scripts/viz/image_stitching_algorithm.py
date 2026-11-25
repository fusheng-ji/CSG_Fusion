#!/usr/bin/env python3
"""Stitch the ScanNet++ renders of several methods into one comparison image per test sample.

Reads the images saved by the uniform baseline harness:

    python -m scripts.eval.baselines --method <m> --checkpoint <ckpt> --save-images
    -> results/baselines/<m>/scannetpp/overlap_<o>/images/<scene>/<scene>_<target>_<ctx1>_<ctx2>/
           target_pred.png  target_gt.png  mask.png  context_<j>_pred.png

The harness does not save the context input views, so they are read from the ScanNet++ data (--data_root) with
the same crop / resize as the test set. Only samples rendered by every method are stitched.

Layout (left to right): the two context views stacked in one column, one render per method (in --methods
order), the ground-truth target view. Output: <output_dir>/overlap_<o>/<target>_<ctx1>_<ctx2>_<scene>.png
Paths are relative to the repository root.
"""

# --- path bootstrap: run from anywhere, resolve paths relative to the repo root ---
import os as _os
import sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
# --- end bootstrap ---

import argparse
import os
import sys

from PIL import Image
from tqdm import tqdm

DEFAULT_METHODS = ["pixelsplat", "mvsplat", "noposplat", "splatt3r", "csg_fusion"]
TRANSPARENT_FILL = (255, 255, 255, 0)
GAP = 20  # pixels between the images


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--overlap", type=float, required=True, help="overlap level (0.3, 0.5, 0.7 or 0.9)")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        help="methods, in display order (default: %(default)s)")
    parser.add_argument("--results_root", default="results/baselines",
                        help="harness output root; images are read from "
                             "<results_root>/<method>/scannetpp/overlap_<o>/images (default: %(default)s)")
    parser.add_argument("--results_paths", nargs="+", default=None,
                        help="explicit images/ folder per method (same order as --methods), instead of "
                             "--results_root")
    parser.add_argument("--data_root", default="data/scannetpp",
                        help="ScanNet++ root, for the context input views (default: %(default)s)")
    parser.add_argument("--no_context", action="store_true",
                        help="leave out the context views (no ScanNet++ data needed)")
    parser.add_argument("--output_dir", default="results/comparison/scannetpp",
                        help="output root; images go to <output_dir>/overlap_<o>/ (default: %(default)s)")
    parser.add_argument("--target_size", type=int, default=512, help="size of the square renders (default: 512)")
    parser.add_argument("--max_samples", type=int, default=None, help="stitch at most this many samples")

    args = parser.parse_args()
    if args.results_paths is not None and len(args.results_paths) != len(args.methods):
        parser.error("--results_paths needs one folder per method in --methods")
    return args


def list_samples(images_dir):
    """{(scene, target, ctx1, ctx2): sample dir} of every sample with a target render in a harness images/ dir."""
    samples = {}
    if not os.path.isdir(images_dir):
        return samples
    for scene in sorted(os.listdir(images_dir)):
        scene_dir = os.path.join(images_dir, scene)
        if not os.path.isdir(scene_dir):
            continue
        for name in sorted(os.listdir(scene_dir)):
            sample_dir = os.path.join(scene_dir, name)
            if not name.startswith(scene + "_") or not os.path.isfile(os.path.join(sample_dir, "target_pred.png")):
                continue
            ids = name[len(scene) + 1:].split("_")  # <target>_<ctx1>_<ctx2>
            if len(ids) != 3:
                continue
            samples[(scene, *ids)] = sample_dir
    return samples


class ContextViews:
    """The context input views of the ScanNet++ test set, cropped and resized as in the evaluation."""

    def __init__(self, root, resolution):
        for p in ("src/pixelsplat_src", "src/mast3r_src", "src/mast3r_src/dust3r"):
            if p not in sys.path:
                sys.path.append(p)
        import data.scannetpp.scannetpp as scannetpp
        self.data = scannetpp.ScanNetPPData(root, "val")
        self.resolution = [resolution, resolution]

    def __call__(self, scene, index):
        try:
            img = self.data.get_view(scene, int(index), self.resolution)["original_img"]
            return img.convert("RGBA") if isinstance(img, Image.Image) else Image.fromarray(img).convert("RGBA")
        except Exception as e:  # scene / frame not in the downloaded data
            print(f"Could not load context view {scene}/{index}: {e}")
            return None


def load_image(path, size):
    try:
        return Image.open(path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Could not load image {path}: {e}")
        return Image.new("RGBA", (size, size), TRANSPARENT_FILL)


def stitch_images(context_views, method_paths, target_path, target_size):
    all_images = []

    # The two context views, stacked in one column as high as the renders
    if context_views is not None:
        ctx_size = (target_size - GAP) // 2
        column = Image.new("RGBA", (ctx_size, target_size), TRANSPARENT_FILL)
        for i, view in enumerate(context_views):
            if view is not None:
                view = view.resize((ctx_size, ctx_size), Image.Resampling.LANCZOS)
                column.paste(view, (0, i * (ctx_size + GAP)), view)
        all_images.append(column)

    # One render per method, then the ground-truth target view
    all_images += [load_image(p, target_size) for p in method_paths]
    all_images.append(load_image(target_path, target_size))

    total_width = sum(img.width for img in all_images) + GAP * (len(all_images) - 1)
    max_height = max(img.height for img in all_images)
    final = Image.new("RGBA", (total_width, max_height), TRANSPARENT_FILL)
    x_offset = 0
    for img in all_images:
        final.paste(img, (x_offset, (max_height - img.height) // 2), img)
        x_offset += img.width + GAP
    return final


def main():
    args = parse_arguments()
    overlap_tag = f"overlap_{args.overlap}"

    if args.results_paths is None:
        images_dirs = [os.path.join(args.results_root, m, "scannetpp", overlap_tag, "images") for m in args.methods]
    else:
        images_dirs = args.results_paths

    per_method = []
    for method, images_dir in zip(args.methods, images_dirs):
        samples = list_samples(images_dir)
        print(f"{method}: {len(samples)} samples in {images_dir}")
        if not samples:
            print(f"  (run `python -m scripts.eval.baselines --method {method} --checkpoint <ckpt> --save-images`)")
        per_method.append(samples)

    common = sorted(set.intersection(*(set(s) for s in per_method)))
    print(f"{len(common)} samples rendered by every method")
    if args.max_samples:
        common = common[:args.max_samples]
    if not common:
        return

    context_views = None if args.no_context else ContextViews(args.data_root, args.target_size)

    save_dir = os.path.join(args.output_dir, overlap_tag)
    os.makedirs(save_dir, exist_ok=True)

    for key in tqdm(common, desc="Stitching"):
        scene, target, ctx1, ctx2 = key
        method_paths = [os.path.join(s[key], "target_pred.png") for s in per_method]
        target_path = os.path.join(per_method[-1][key], "target_gt.png")
        views = None if context_views is None else [context_views(scene, ctx1), context_views(scene, ctx2)]
        stitched = stitch_images(views, method_paths, target_path, args.target_size)
        stitched.save(os.path.join(save_dir, f"{target}_{ctx1}_{ctx2}_{scene}.png"))

    print(f"Stitched {len(common)} samples into {save_dir}")


if __name__ == "__main__":
    main()
