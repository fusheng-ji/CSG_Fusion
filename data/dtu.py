import json
from pathlib import Path
from typing import List, Dict
from jaxtyping import Float, UInt8
from einops import repeat, rearrange
from torch import Tensor
from io import BytesIO
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as T

class DTUDataset(data.Dataset):
    def __init__(
        self,
        root_dir: str,
        index_json: str,
        evaluation_index_json: str,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.index_json = index_json
        self.evaluation_index_json = evaluation_index_json
        self.to_tensor = T.ToTensor()

        # Load the evaluation index
        with open(evaluation_index_json, "r") as f:
            self.eval_idx = json.load(f)

        # Build the sample list
        self.samples: List[Dict] = []
        for scene_index_name, cfg in self.eval_idx.items():
            base_scene = "_".join(scene_index_name.split("_")[:-1])
            scene_name = base_scene.replace("_train", "")
            ctx_idx: List[int] = cfg["context"]
            tgt_idx: List[int] = (
                cfg["target"] if isinstance(cfg["target"], list) else [cfg["target"]]
            )
            for t_idx in tgt_idx:
                self.samples.append(
                    {
                        "case_name": scene_name,
                        "scene_name": base_scene,
                        "context_indices": ctx_idx,
                        "target_index": t_idx,
                    }
                )
        with open(index_json, "r") as f:
            index = json.load(f)
        self.index = {k: Path(self.root_dir / 'test' / v) for k, v in index.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        chunk = torch.load(str(self.index[sample["scene_name"]]), weights_only=True)
        for item in chunk:
            if item["key"] == sample["scene_name"]:
                scene_data = item
                break
        context_images = [scene_data["images"][i] for i in sample["context_indices"]]
        target_image = scene_data["images"][sample["target_index"]]
        context_cameras = [scene_data["cameras"][i] for i in sample["context_indices"]]
        target_camera = scene_data["cameras"][sample["target_index"]]
        context_extrinsics = []
        context_intrinsics = []
        # Hand-set constant (like focal_scale in convert_poses), not derived from the DTU calibration. Applies to every
        # method evaluated through this loader.
        scale = 0.38
        for camera in context_cameras:
            extrinsics, intrinsics = self.convert_poses(camera.unsqueeze(0))
            extrinsics[:, :3, 3] *= scale
            context_extrinsics.append(extrinsics)
            context_intrinsics.append(intrinsics)
        context_extrinsics = torch.stack(context_extrinsics)
        context_intrinsics = torch.stack(context_intrinsics)

        target_extrinsics, target_intrinsics = self.convert_poses(target_camera.unsqueeze(0))
        target_extrinsics[:, :3, 3] *= scale

        context_images = self.convert_images(context_images)
        target_image = self.convert_images([target_image])


        context_images, context_intrinsics = self.center_crop(context_images, context_intrinsics, (512, 512))
        target_image, target_intrinsics = self.center_crop(target_image, target_intrinsics, (512, 512))

        # Build the output dict
        def make_entry(img,  pose, K, v_idx, typ):
            return {
                "img": img * 2 - 1,       # [-1, 1], as MASt3R expects (ImgNorm in the ScanNet++ loader)
                "original_img": img,
                "camera_pose": pose,
                "camera_intrinsics": K,
                "depthmap": torch.ones((1, 512, 512), dtype=torch.float32),
                "valid_mask": torch.ones((1, 512, 512), dtype=torch.bool),
                "dataset": "dtu",
                "label": f"dtu/{sample['scene_name']}",
                "instance": f"{sample['case_name']}_{v_idx}",
                "is_metric_scale": True,
            }
        context_entries = []
        for i in range(2):
            context_entries.append(make_entry(context_images[i], context_extrinsics[i].squeeze(0), context_intrinsics[i].squeeze(0), sample["context_indices"][i], "context"))

        target_entries = [make_entry(target_image.squeeze(0), target_extrinsics.squeeze(0), target_intrinsics.squeeze(0), sample["target_index"], "target")]

        return {
            "context": context_entries,
            "target": target_entries,
            "scene": sample['scene_name'],
            "scene_name": sample["scene_name"],
            "context_indices": sample["context_indices"],
            "target_index": sample["target_index"],
        }

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Recover pixel intrinsics from normalized intrinsics
        saved_fx, saved_fy, saved_cx, saved_cy = poses[:, :4].T

        original_width = 1600.0
        original_height = 1200.0
        current_width = 640.0
        current_height = 512.0

        # Original-resolution intrinsics
        fx_original = saved_fx * original_width
        fy_original = saved_fy * original_height
        cx_original = saved_cx * original_width   # 0.5 * 1600 = 800
        cy_original = saved_cy * original_height  # 0.5 * 1200 = 600

        # Rescale intrinsics to the current image size (640x512)
        scale_x = current_width / original_width   # 640/1600 = 0.4
        scale_y = current_height / original_height # 512/1200 = 0.4267

        # Hand-set constant, see `scale` above
        focal_scale = 1.23  # focal length multiplier

        fx = fx_original * scale_x * focal_scale
        fy = fy_original * scale_y * focal_scale
        cx = cx_original * scale_x
        cy = cy_original * scale_y

        # Build the 3x3 intrinsics matrix
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        intrinsics[:, 0, 0] = fx   # fx
        intrinsics[:, 1, 1] = fy   # fy
        intrinsics[:, 0, 2] = cx   # cx
        intrinsics[:, 1, 2] = cy   # cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)


    def center_crop(
        self,
        images: Float[Tensor, "*#batch c h w"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
        shape: tuple[int, int],
    ) -> tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    ]:
        *_, h_in, w_in = images.shape
        h_out, w_out = shape

        # Note that odd input dimensions induce half-pixel misalignments.
        row = (h_in - h_out) // 2
        col = (w_in - w_out) // 2

        # Center-crop the image.
        images = images[..., :, row : row + h_out, col : col + w_out]

        # Adjust the intrinsics to account for the cropping.
        intrinsics = intrinsics.clone()
        intrinsics[..., 0, 2] -= col  # cx
        intrinsics[..., 1, 2] -= row  # cy
        return images, intrinsics
