"""Build the processed ScanNet++ DSLR layout CSG-Fusion trains on, from the official v1 download.

Input  (official download, data_root = data/scannetpp/raw):
    raw/data/<scene>/dslr/{colmap/, nerfstudio/transforms.json, nerfstudio/transforms_undistorted.json,
                           resized_undistorted_images/, train_test_lists.json}
    raw/data/<scene>/scans/mesh_aligned_0.05.ply
Output (what data/scannetpp/scannetpp.py reads):
    processed/<scene>/dslr/undistorted_images              -> symlink to the official resized_undistorted_images
    processed/<scene>/dslr/undistorted_depths/<frame>.png  uint16, millimetres, 0 = invalid
    processed/<scene>/dslr/nerfstudio/transforms_undistorted.json

Depth follows the Splatt3R / SplaTAM ScanNet++ pipeline: render the mesh in the distorted OPENCV_FISHEYE
DSLR camera with renderpy (https://github.com/liu115/renderpy, near 0.05 m, far 20 m), then undistort with
cv2.fisheye maps and nearest-neighbour interpolation. The undistorted intrinsics are taken from the official
transforms_undistorted.json, so the depth maps are pixel-aligned with the official undistorted images.

usage: python data/scannetpp/prepare_processed.py [--scenes a b ...] [--root data/scannetpp]
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_colmap_text(colmap_dir):
    """Single camera + world_to_camera (4x4) per image name, from COLMAP text files."""
    cams = [l.split() for l in open(colmap_dir / 'cameras.txt') if l.strip() and not l.startswith('#')]
    assert len(cams) == 1, 'multiple cameras not supported'
    _, model, w, h, *params = cams[0]
    lines = [l for l in open(colmap_dir / 'images.txt') if not l.startswith('#')]
    images = {}
    for line in lines[0::2]:                     # every other line holds the 2D points
        if not line.strip():
            continue
        e = line.split()
        T = np.eye(4)
        T[:3, :3] = qvec2rotmat(np.array(e[1:5], dtype=float))
        T[:3, 3] = np.array(e[5:8], dtype=float)
        images[e[9]] = T
    return model, int(w), int(h), np.array(params, dtype=float), images


def process_scene(root, scene, renderer_cls, near=0.05, far=20.0, overwrite=False):
    raw = root / 'raw' / 'data' / scene
    out = root / 'processed' / scene / 'dslr'
    model, W, H, params, images = read_colmap_text(raw / 'dslr' / 'colmap')
    assert model == 'OPENCV_FISHEYE', model
    depth_dir = out / 'undistorted_depths'
    todo = [n for n in sorted(images) if overwrite or not (depth_dir / n.replace('.JPG', '.png')).exists()]

    # undistortion maps: distorted fisheye camera -> official undistorted pinhole camera
    t = json.load(open(raw / 'dslr' / 'nerfstudio' / 'transforms.json'))
    u = json.load(open(raw / 'dslr' / 'nerfstudio' / 'transforms_undistorted.json'))
    K = np.array([[t['fl_x'], 0, t['cx']], [0, t['fl_y'], t['cy']], [0, 0, 1]])
    D = np.array([t['k1'], t['k2'], t['k3'], t['k4']])
    new_K = np.array([[u['fl_x'], 0, u['cx']], [0, u['fl_y'], u['cy']], [0, 0, 1]])
    assert (int(t['w']), int(t['h'])) == (W, H) == (int(u['w']), int(u['h']))
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (W, H), cv2.CV_32FC1)

    if todo:
        depth_dir.mkdir(parents=True, exist_ok=True)
        render = renderer_cls()
        render.setupMesh(str(raw / 'scans' / 'mesh_aligned_0.05.ply'))
        fx, fy, cx, cy = params[:4]
        render.setupCamera(H, W, fx, fy, cx, cy, model, params[4:])
        for name in tqdm(todo, desc=scene, leave=False):
            _, depth, _ = render.renderAll(images[name], near, far)
            depth = (depth.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)
            depth = cv2.remap(depth, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(str(depth_dir / name.replace('.JPG', '.png')), depth)

    img_link = out / 'undistorted_images'
    if not img_link.exists():
        img_link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(raw / 'dslr' / 'resized_undistorted_images', out), img_link)
    tf = out / 'nerfstudio' / 'transforms_undistorted.json'
    if not tf.exists():
        tf.parent.mkdir(parents=True, exist_ok=True)
        tf.write_text((raw / 'dslr' / 'nerfstudio' / 'transforms_undistorted.json').read_text())
    return len(todo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=str(Path(__file__).resolve().parent))
    p.add_argument('--scenes', nargs='*', help='default: every scene with a coverage file')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    root = Path(args.root)
    scenes = args.scenes or sorted(f[:-5] for f in os.listdir(root / 'coverage') if f.endswith('.json'))
    import renderpy
    done = 0
    for scene in tqdm(scenes, desc='scenes'):
        if not (root / 'raw' / 'data' / scene / 'scans' / 'mesh_aligned_0.05.ply').exists():
            print(f'skip {scene}: not downloaded yet')
            continue
        done += process_scene(root, scene, renderpy.Render, overwrite=args.overwrite)
    print(f'rendered {done} depth maps')


if __name__ == '__main__':
    main()
