---
license: cc-by-nc-sa-4.0
library_name: pytorch
pipeline_tag: image-to-3d
base_model:
  - naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
  - facebook/VGGT-1B
tags:
  - gaussian-splatting
  - 3d-gaussian-splatting
  - 3d-reconstruction
  - novel-view-synthesis
  - feed-forward
  - pose-free
  - sparse-view
  - mast3r
  - vggt
  - splatt3r
datasets:
  - scannetpp
metrics:
  - psnr
  - ssim
  - lpips
---

# CSG-Fusion: Consistent Sparse-View Gaussian Splatting via Matching-based Fusion

Pretrained weights of **CSG-Fusion** (ICCV 2025 Workshops, E2E3D, **Best Paper Award**), a feed-forward
**3D Gaussian Splatting** model that reconstructs a scene from **two unposed, uncalibrated RGB images** in a single
forward pass. It builds on MASt3R / Splatt3R and removes the duplicated Gaussians of overlapping views with a
**Matching-based Gaussian Filter** and a **Mask-based Feature Aggregation**, giving fewer, consistent Gaussians and
better novel views.

**Yan Xia**\* · **Wenbo Ji**\* · **Weirong Chen** · **Daniel Cremers** (USTC, TUM, MCML; \* equal contribution)

- **Code:** https://github.com/fusheng-ji/CSG_Fusion
- **Project page:** https://fusheng-ji.github.io/CSG_Fusion/
- **Paper:** [CVF open access](https://openaccess.thecvf.com/content/ICCV2025W/E2E3D/papers/Xia_CSG-Fusion_Consistent_Sparse-View_Gaussian_Splatting_via_Matching-based_Fusion_ICCVW_2025_paper.pdf) · DOI [10.1109/ICCVW69036.2025.00276](https://doi.org/10.1109/ICCVW69036.2025.00276)

## Files

| File | Content |
|---|---|
| `csg_fusion.ckpt` (3.0 GB) | The published model (MASt3R ViT-L backbone, `configs/csg_fusion.yaml`): weights and training config, as read by `main.load_model` |
| `csg_fusion_vggt.ckpt` (4.2 GB) | CSG-Fusion with the frozen MASt3R replaced by a frozen VGGT-1B aggregator (`configs/csg_fusion_vggt.yaml`); all weights included, so VGGT-1B is not downloaded |
| `LICENSE`, `MASt3R_CHECKPOINTS_NOTICE`, `VGGT_LICENSE.txt` | CC BY-NC-SA 4.0, and the terms of the MASt3R and VGGT checkpoints |

## Usage

```bash
git clone https://github.com/fusheng-ji/CSG_Fusion.git && cd CSG_Fusion
git submodule update --init --recursive third_party/diff-gaussian-rasterization
conda env create -f environment.yml && conda activate csg_fusion
bash scripts/install_rasterizers.sh
python demo.py --image1 a.jpg --image2 b.jpg --out demo_out/   # downloads csg_fusion.ckpt
python demo.py --vggt --mode joint --images a.jpg b.jpg c.jpg --out demo_out/   # csg_fusion_vggt.ckpt, one VGGT pass over all images
```

It needs an NVIDIA GPU with about 6 GB of memory; GPUs that need CUDA ≥ 12.8 use `environment-cu128.yml`.

```bash
huggingface-cli download fushengji/CSG-Fusion csg_fusion.ckpt --local-dir checkpoint     # for load_model below
```

```python
import main
model = main.load_model('checkpoint/csg_fusion.ckpt', 'cuda').eval()
```

- **Input:** two RGB images of the same static scene (centre-cropped and resized to 512×512). No poses, no intrinsics.
- **Output:** metric-scale 3D Gaussians in the camera frame of the first image (positions, covariances, colours,
  opacities); `utils/export.save_as_ply` writes them as a standard 3DGS `.ply` file.

## Results (ScanNet++, uniform evaluation, PSNR↑ / SSIM↑ / LPIPS↓ at 256×256)

| View overlap | 0.9 | 0.7 | 0.5 | 0.3 |
|---|---|---|---|---|
| Splatt3R | 17.84 / 0.676 / 0.234 | 18.73 / 0.716 / 0.196 | 19.25 / 0.746 / 0.183 | 19.18 / 0.772 / 0.184 |
| **CSG-Fusion** | **20.38 / 0.733 / 0.188** | **20.47 / 0.749 / 0.173** | **20.57 / 0.770 / 0.166** | **20.32 / 0.791 / 0.170** |
| CSG-Fusion (VGGT) | 21.21 / 0.750 / 0.173 | 20.64 / 0.749 / 0.176 | 20.49 / 0.761 / 0.176 | 20.25 / 0.783 / 0.178 |

Same test pairs, mask and metric code for all three. See the repository README for the paper's table and the comparison with nine feed-forward baselines.

## Training data and license

Both models are trained on ScanNet++ v1 (DSLR): `csg_fusion.ckpt` is fine-tuned from the MASt3R checkpoint
`MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`, `csg_fusion_vggt.ckpt` contains the weights of
[facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) (CC BY-NC 4.0). Released under CC BY-NC-SA 4.0 for
**non-commercial use only**; the MASt3R checkpoint terms (`MASt3R_CHECKPOINTS_NOTICE`), the VGGT license
(`VGGT_LICENSE.txt`) and the ScanNet++ terms of use also apply.

## Citation

```bibtex
@InProceedings{Xia_2025_ICCV,
    author    = {Xia, Yan and Ji, Wenbo and Chen, Weirong and Cremers, Daniel},
    title     = {CSG-Fusion: Consistent Sparse-View Gaussian Splatting via Matching-based Fusion},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {2653-2662},
    doi       = {10.1109/ICCVW69036.2025.00276}
}
```
