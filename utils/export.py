"""Saving model outputs: fused Gaussians as standard 3DGS .ply files, renders, and Lightning callbacks that
write inputs / renders / Gaussians of some (training) or all (DTU evaluation) batches to disk."""
import json
import os

import lightning as L
import numpy as np
import torch
import torchvision
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


class SaveBatchData(L.Callback):
    '''A Lightning callback that occasionally saves batch inputs and outputs to disk.
    It is not critical to the training process, and can be disabled if unwanted.'''

    def __init__(self, save_dir, train_save_interval=100, val_save_interval=100, test_save_interval=100):
        self.save_dir = save_dir
        self.train_save_interval = train_save_interval
        self.val_save_interval = val_save_interval
        self.test_save_interval = test_save_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.train_save_interval == 0 and trainer.global_rank == 0:
            self.save_batch_data('train', trainer, pl_module, batch, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.val_save_interval == 0 and trainer.global_rank == 0:
            self.save_batch_data('val', trainer, pl_module, batch, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.test_save_interval == 0 and trainer.global_rank == 0:
            self.save_batch_data('test', trainer, pl_module, batch, batch_idx)

    def save_batch_data(self, prefix, trainer, pl_module, batch, batch_idx):

        print(f'Saving {prefix} data at epoch {trainer.current_epoch} and batch {batch_idx}')

        # Run the batch through the model again
        view1, view2 = batch['context']
        pred1, pred2 = pl_module.forward(view1, view2)
        color, context_color, *_ = pl_module.decoder(batch, pred1, pred2)


        # Save the data
        save_dir = os.path.join(
            self.save_dir,
            f"{prefix}_epoch_{trainer.current_epoch}_batch_{batch_idx}"
        )
        log_batch_files(batch, color, context_color, pred1, pred2, save_dir)


class SaveAllTestBatchData(L.Callback):
    '''Lightning callback that saves inputs, outputs and metrics of every test batch to disk.
    Handles test_step returning a (loss, psnr, lpips, ssim) tuple.'''

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.batch_counter = 0
        self.metrics_log = {}

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.global_rank == 0:
            # Extract metrics from outputs
            batch_metrics = self.extract_metrics(outputs, batch_idx)
            self.metrics_log[f"batch_{batch_idx}"] = batch_metrics

            # Save batch data
            self.save_batch_data('test', trainer, pl_module, batch, batch_idx, batch_metrics)
            self.batch_counter += 1

    def on_test_end(self, trainer, pl_module):
        """Save metrics of all batches at the end of testing."""
        if trainer.global_rank == 0 and self.metrics_log:
            metrics_path = os.path.join(self.save_dir, "all_batch_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_log, f, indent=4)

            print(f"Saved metrics for all batches to {metrics_path}")

    def extract_metrics(self, outputs, batch_idx):
        """Extract metrics from the (loss, psnr, lpips, ssim) tuple returned by test_step."""
        metrics = {}

        # Tuple return value
        if isinstance(outputs, tuple) and len(outputs) >= 4:
            metrics["loss"] = self._tensor_to_scalar(outputs[0])
            metrics["psnr"] = self._tensor_to_scalar(outputs[1])
            metrics["lpips"] = self._tensor_to_scalar(outputs[2])
            metrics["ssim"] = self._tensor_to_scalar(outputs[3])
        # Dict return value
        elif isinstance(outputs, dict):
            for key, value in outputs.items():
                metrics[key] = self._tensor_to_scalar(value)
        # Single tensor return value
        elif isinstance(outputs, torch.Tensor):
            metrics["loss"] = self._tensor_to_scalar(outputs)

        # Add the batch index
        metrics["batch_idx"] = batch_idx

        return metrics

    def _tensor_to_scalar(self, value):
        """Convert a tensor to a Python scalar."""
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.detach().cpu().numpy().tolist()
        return value

    def save_batch_data(self, prefix, trainer, pl_module, batch, batch_idx, metrics):
        print(f'Saving batch {batch_idx} data and metrics')

        # Run the batch through the model again
        view1, view2 = batch['context']
        pred1, pred2 = pl_module.forward(view1, view2)
        color, context_color, *_ = pl_module.decoder(batch, pred1, pred2)


        # Save data
        save_dir = os.path.join(
            self.save_dir,
            f"{prefix}_batch_{batch_idx}"
        )
        os.makedirs(save_dir, exist_ok=True)

        # Save batch metrics to a json file
        metrics_path = os.path.join(save_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        # Save visualizations
        log_batch_files(batch, color, context_color, pred1, pred2, save_dir)


def gaussians_to_numpy(pred1, pred2, batch_index=0, keep_filtered=False):
    """The fused Gaussians of one batch item as numpy arrays (N = view-1 pixels + kept view-2 pixels).

    View-1 contributes one Gaussian per pixel. Of view 2, only the Gaussians kept by the Matching-based Gaussian
    Filter (``pred2['mask']``) are returned, unless ``keep_filtered``; the filtered ones already have opacity 0.
    ``pred2`` may also be a list of such predictions (multi-view inference, src/multiview.py: views 2..N).
    Positions are in the camera frame of view 1 (OpenCV convention: x right, y down, z forward), in metres.

    Returns a dict with ``means`` (N, 3), ``covariances`` (N, 3, 3), ``sh`` (N, 3, n_sh) and ``opacities`` (N,),
    opacities after the sigmoid.
    """
    others = list(pred2) if isinstance(pred2, (list, tuple)) else [pred2]
    out = {}
    for key, key2 in (('means', 'means_in_other_view'), ('covariances', 'covariances'), ('sh', 'sh'),
                      ('opacities', 'opacities')):
        parts = [pred1[key][batch_index].flatten(0, 1)]           # (H*W, ...)
        for pred in others:
            keep = pred['mask'][batch_index].reshape(-1).bool()
            if keep_filtered:
                keep = torch.ones_like(keep)
            parts.append(pred[key2][batch_index].flatten(0, 1)[keep])
        out[key] = torch.cat(parts).detach().float().cpu().numpy()
    out['opacities'] = out['opacities'].reshape(-1)
    return out


def covariance_to_scale_rotation(covariances):
    """Decompose symmetric covariances as R diag(s^2) R^T.

    Returns ``scales`` (N, 3) and unit quaternions (N, 4) in the (w, x, y, z) order of 3DGS .ply files.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariances.astype(np.float64))   # columns = principal axes
    # A rotation needs det = +1; flipping one axis leaves R diag(s^2) R^T unchanged.
    flip = np.linalg.det(eigenvectors) < 0
    eigenvectors[flip, :, 0] *= -1
    scales = np.sqrt(np.clip(eigenvalues, 1e-20, None))
    xyzw = Rotation.from_matrix(eigenvectors).as_quat()      # scipy order (x, y, z, w)
    return scales.astype(np.float32), np.roll(xyzw, 1, axis=-1).astype(np.float32)


def save_as_ply(pred1, pred2, save_path, batch_index=0, keep_filtered=False):
    """Write the fused Gaussians of one batch item as a standard 3D Gaussian Splatting .ply file (``pred2``: one
    prediction or, for multi-view inference, the list of views 2..N).

    The file follows the attribute layout of the original 3DGS code (x, y, z, nx, ny, nz, f_dc_*, f_rest_*,
    opacity as a logit, scale_* as log scales, rot_* as a (w, x, y, z) quaternion), so it opens in the usual
    viewers (e.g. SuperSplat, the 3DGS SIBR viewer). Only the DC term of the colour is written: the published
    model predicts view-independent colour (one SH coefficient per channel).
    """
    g = gaussians_to_numpy(pred1, pred2, batch_index, keep_filtered)
    # Decompose the covariances the renderer uses rather than re-using the predicted scale / quaternion: the
    # quaternion-to-matrix conversion adds an epsilon to |q|^2, so for small |q| those two do not exactly
    # reproduce the rendered covariance.
    scales, rotations = covariance_to_scale_rotation(g['covariances'])
    opacity = np.clip(g['opacities'], 1e-6, 1 - 1e-6)
    attributes = {
        'x': g['means'][:, 0], 'y': g['means'][:, 1], 'z': g['means'][:, 2],
        'nx': 0.0, 'ny': 0.0, 'nz': 0.0,
        'f_dc_0': g['sh'][:, 0, 0], 'f_dc_1': g['sh'][:, 1, 0], 'f_dc_2': g['sh'][:, 2, 0],
        'opacity': np.log(opacity / (1 - opacity)),
        'scale_0': np.log(scales[:, 0]), 'scale_1': np.log(scales[:, 1]), 'scale_2': np.log(scales[:, 2]),
        'rot_0': rotations[:, 0], 'rot_1': rotations[:, 1], 'rot_2': rotations[:, 2], 'rot_3': rotations[:, 3],
    }
    elements = np.empty(len(g['means']), dtype=[(name, 'f4') for name in attributes])
    for name, value in attributes.items():
        elements[name] = value
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    PlyData([PlyElement.describe(elements, 'vertex')]).write(save_path)
    return len(elements)


@torch.no_grad()
def log_batch_files(batch, color, context_color, pred1, pred2, save_dir):
    """Save the inputs, renders and fused Gaussians (.ply, first batch item) of a batch for inspection."""
    os.makedirs(save_dir, exist_ok=True)

    save_as_ply(pred1, pred2, os.path.join(save_dir, "gaussians.ply"))

    context_images = torch.stack([view["img"] for view in batch["context"]], dim=1) / 2 + 0.5      # [-1, 1] -> [0, 1]
    context_original_images = torch.stack([view["original_img"] for view in batch["context"]], dim=1)
    for b in range(min(context_images.shape[0], 4)):
        torchvision.utils.save_image(context_images[b], os.path.join(save_dir, f"sample_{b}_img_context.jpg"))
        torchvision.utils.save_image(context_original_images[b],
                                     os.path.join(save_dir, f"sample_{b}_original_img_context.jpg"))

    target_original_images = torch.stack([view["original_img"] for view in batch["target"]], dim=1)
    for b in range(min(target_original_images.shape[0], 4)):
        torchvision.utils.save_image(target_original_images[b],
                                     os.path.join(save_dir, f"sample_{b}_original_img_target.jpg"))

    for b in range(min(color.shape[0], 4)):
        torchvision.utils.save_image(color[b, ...], os.path.join(save_dir, f"sample_{b}_rendered_color.jpg"))
    if context_color is not None:
        for b in range(min(context_color.shape[0], 4)):
            torchvision.utils.save_image(context_color[b, ...],
                                         os.path.join(save_dir, f"sample_{b}_rendered_context_color.jpg"))
