"""CSG-Fusion demo: two (or more) images in, one fused set of 3D Gaussians out.

    python demo.py --image1 a.jpg --image2 b.jpg --out demo_out/
    python demo.py --images a.jpg b.jpg c.jpg d.jpg --out demo_out/      # multi-view inference (src/multiview.py)

Writes to --out:
    gaussians.ply      the fused 3D Gaussians (standard 3DGS layout; opens in SuperSplat, the SIBR viewer, ...)
    input_{i}.png      the inputs after the 512x512 centre crop the model works on
    render_{i}.png     the Gaussians re-rendered at the input cameras
    trajectory.mp4     a fly-through from camera 1 through the other cameras (trajectory_*.png frames if no ffmpeg)
    cameras.json       the estimated focal length and the camera poses (camera 1 is the world frame)

No camera poses or intrinsics are needed. The model predicts metric 3D points in the frame of image 1; for
rendering, this script estimates a shared focal length from those points and the pose of every other image with
PnP, which assumes all images come from the same camera. With more than two images the two-view model is run
without retraining (--mode pairwise: on pairs chained from image 1 over the best-matching images; --mode joint:
one VGGT pass over all images, --vggt only) and --filter decides which duplicates are removed (see src/multiview.py). The checkpoint is downloaded from
the Hugging Face Hub unless it exists locally.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import main  # noqa: E402
import utils.export as export  # noqa: E402
from src.multiview import predict_multiview  # noqa: E402
from src.mast3r_src.dust3r.dust3r.datasets.utils.transforms import ImgNorm  # noqa: E402
from src.pixelsplat_src.decoder_splatting_cuda import _get_pose_from_pnp  # noqa: E402

HF_REPO = 'fushengji/CSG-Fusion'   # holds both checkpoints
CHECKPOINTS = ('checkpoint/csg_fusion.ckpt', 'checkpoint/csg_fusion_vggt.ckpt')
DEFAULT_CHECKPOINT = 'checkpoint/csg_fusion.ckpt'
RESOLUTION = 512  # the published model works on 512x512 inputs (16-pixel MASt3R patches, 32x32 tokens)


def resolve_checkpoint(path):
    """A local checkpoint, or a published one from the Hugging Face Hub (cached after the first download)."""
    if os.path.isfile(path):
        return path
    if path not in CHECKPOINTS:
        sys.exit(f'Checkpoint {path} not found')
    from huggingface_hub import hf_hub_download
    name = os.path.basename(path)
    print(f'Downloading {name} from huggingface.co/{HF_REPO} ...')
    return hf_hub_download(repo_id=HF_REPO, filename=name)


def load_view(path, device):
    """Centre-crop to a square, resize to 512x512, and build the view dict the model expects."""
    image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')    # phone photos store their orientation in EXIF
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
    return {
        'img': ImgNorm(image)[None].to(device),                                     # [-1, 1], encoder input
        'original_img': torchvision.transforms.functional.to_tensor(image)[None].to(device),  # [0, 1], colours
        'true_shape': torch.tensor([[RESOLUTION, RESOLUTION]], device=device),
        'instance': [os.path.abspath(path)],  # MASt3R's encoder uses it to detect symmetrised batches
    }


def estimate_focal(pts3d, conf=None):
    """Least-squares focal length (pixels) of a centred pinhole camera that projects pts3d (H, W, 3) to its grid."""
    h, w = pts3d.shape[:2]
    v, u = torch.meshgrid(torch.arange(h, device=pts3d.device), torch.arange(w, device=pts3d.device), indexing='ij')
    uv = torch.stack([u - (w - 1) / 2, v - (h - 1) / 2], dim=-1).float()        # pixel offsets from the centre
    xy = pts3d[..., :2] / pts3d[..., 2:3].clamp(min=1e-6)                       # normalised image coordinates
    valid = pts3d[..., 2] > 0
    if conf is not None:
        valid &= conf >= conf[valid].median()                                   # the more confident half
    xy, uv = xy[valid], uv[valid]
    return float((xy * uv).sum() / (xy * xy).sum())                             # argmin_f |f * xy - uv|^2


def trajectory(poses, n):
    """About n camera-to-world poses through ``poses`` in order (each segment interpolated)."""
    per = max(2, n // max(1, len(poses) - 1))
    out = []
    for a, b in zip(poses[:-1], poses[1:]):
        out.extend(interpolate_poses(a, b, per)[:-1])
    return out + [poses[-1]]


def interpolate_poses(pose_a, pose_b, n):
    """n camera-to-world poses from pose_a to pose_b (slerp on rotation, linear on position)."""
    from scipy.spatial.transform import Rotation, Slerp
    rotations = Rotation.from_matrix(np.stack([pose_a[:3, :3], pose_b[:3, :3]]))
    slerp = Slerp([0.0, 1.0], rotations)
    poses = []
    for t in np.linspace(0.0, 1.0, n):
        pose = np.eye(4)
        pose[:3, :3] = slerp([t]).as_matrix()[0]
        pose[:3, 3] = (1 - t) * pose_a[:3, 3] + t * pose_b[:3, 3]
        poses.append(pose)
    return poses


@torch.no_grad()
def render(model, preds, intrinsics, poses, device, chunk=8):
    """Render the fused Gaussians (``preds``: reference view first) from camera-to-world `poses` (camera 1 =
    identity), `chunk` views at a time."""
    eye = torch.eye(4, device=device)[None]
    context = [{'camera_pose': eye, 'camera_intrinsics': intrinsics} for _ in preds]
    frames = []
    for start in range(0, len(poses), chunk):
        targets = [{'camera_pose': torch.as_tensor(p, dtype=torch.float32, device=device)[None],
                    'camera_intrinsics': intrinsics,
                    'original_img': torch.zeros(1, 3, RESOLUTION, RESOLUTION, device=device)}
                   for p in poses[start:start + chunk]]
        color, _ = model.decoder.forward_multi({'context': context, 'target': targets}, preds,
                                               render_context=False)
        frames.extend(color[0].clamp(0, 1).cpu())
    return frames


def save_video(frames, out_dir, fps=24):
    images = [(f.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for f in frames]
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(os.path.join(out_dir, 'trajectory.mp4'), images, fps=fps, macro_block_size=1)
    except Exception as error:  # no ffmpeg backend: keep the frames
        print(f'Could not write trajectory.mp4 ({error}); saving the frames as PNG instead')
        for i, image in enumerate(images):
            Image.fromarray(image).save(os.path.join(out_dir, f'trajectory_{i:03d}.png'))


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image1', help='reference image (its camera is the world frame)')
    parser.add_argument('--image2')
    parser.add_argument('--images', nargs='+', help='two or more images, the first one the reference (instead of '
                                                     '--image1 / --image2)')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                        help='local checkpoint; the paths in CHECKPOINTS are downloaded from the Hugging Face Hub if they do not exist')
    parser.add_argument('--vggt', action='store_true', help='the VGGT-backbone model (checkpoint/csg_fusion_vggt.ckpt)')
    parser.add_argument('--out', default='demo_out')
    parser.add_argument('--frames', type=int, default=60, help='frames of the fly-through over the input cameras')
    parser.add_argument('--mode', default=None, choices=['pairwise', 'anchored', 'joint'],
                        help='multi-view inference mode (default: the two-view model for two images, pairwise for more)')
    parser.add_argument('--filter', default=None, choices=['none', 'ref', 'all'],
                        help='duplicate filter (default: the two-view filter for two images, all for more)')
    parser.add_argument('--keep-filtered', action='store_true',
                        help='also write the Gaussians removed by the filter (opacity 0) to the .ply')
    args = parser.parse_args()
    if args.vggt:
        args.checkpoint = 'checkpoint/csg_fusion_vggt.ckpt'
    paths = args.images or [p for p in (args.image1, args.image2) if p]
    if len(paths) < 2:
        parser.error('give --image1 and --image2, or --images with at least two images')
    if not torch.cuda.is_available():
        sys.exit('CSG-Fusion needs a CUDA GPU (the Gaussian rasterizer is a CUDA extension).')
    device = 'cuda'
    os.makedirs(args.out, exist_ok=True)

    model = main.load_model(resolve_checkpoint(args.checkpoint), device).eval()
    views = [load_view(p, device) for p in paths]
    with torch.no_grad():
        if len(views) == 2 and args.mode is None and args.filter is None:
            preds = list(model.forward(views[0], views[1]))
        else:
            preds, _ = predict_multiview(model, views, mode=args.mode or 'pairwise', filter=args.filter or 'all')

    n_view = RESOLUTION * RESOLUTION
    n_kept = [int(p['mask'].sum()) for p in preds[1:]]
    n_written = export.save_as_ply(preds[0], preds[1:], os.path.join(args.out, 'gaussians.ply'),
                                   keep_filtered=args.keep_filtered)
    print(f'Gaussians: {n_view} from view 1 + ' + ' + '.join(f'{k} of view {i}' for i, k in enumerate(n_kept, start=2))
          + f' ({100 * (1 - sum(n_kept) / (n_view * len(n_kept))):.1f}% of views 2..{len(views)} filtered as duplicates);'
          f' {n_written} written')

    # Cameras for rendering: shared focal from the view-1 points, the other poses by PnP on their predicted points.
    focal = estimate_focal(preds[0]['pts3d'][0], preds[0]['conf'][0] if 'conf' in preds[0] else None)
    K = torch.tensor([[focal, 0, (RESOLUTION - 1) / 2], [0, focal, (RESOLUTION - 1) / 2], [0, 0, 1]], device=device)
    grid = torch.from_numpy(np.mgrid[0:RESOLUTION, 0:RESOLUTION].T.reshape(-1, 2)[::4]).float()   # (x, y), row-major
    poses = [np.eye(4)] + [_get_pose_from_pnp(p['pts3d_in_other_view'][0].reshape(-1, 3)[::4], grid, K, device).cpu().numpy()
                           for p in preds[1:]]
    with open(os.path.join(args.out, 'cameras.json'), 'w') as f:
        json.dump({'focal_px': focal, 'resolution': RESOLUTION, 'images': [os.path.abspath(p) for p in paths],
                   'camera_to_world': [pose.tolist() for pose in poses]}, f, indent=2)

    K = K[None]
    for i, (view, pose) in enumerate(zip(views, poses), start=1):
        torchvision.utils.save_image(view['original_img'][0], os.path.join(args.out, f'input_{i}.png'))
        torchvision.utils.save_image(render(model, preds, K, [pose], device)[0], os.path.join(args.out, f'render_{i}.png'))
    save_video(render(model, preds, K, trajectory(poses, args.frames), device), args.out)
    print(f'Saved results to {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main_cli()
