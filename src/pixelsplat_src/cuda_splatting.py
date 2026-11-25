# Adapted from pixelSplat (https://github.com/dcharatan/pixelsplat), MIT License,
# Copyright (c) 2023 David Charatan, Sizhe Li, Andrea Tagliasacchi and Vincent Sitzmann. Modified for CSG-Fusion.
"""Gaussian rasterisation of a batch of views with diff_gaussian_rasterization (pixelSplat's ``render_cuda``)."""
from math import isqrt

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from einops import rearrange

from .projection import cov_upper_triangle, get_fov


def get_projection_matrix(
    near,
    far,
    fov_x,
    fov_y,
):
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render_cuda(
    extrinsics,
    intrinsics,
    near,
    far,
    image_shape: tuple[int, int],
    background_color,
    gaussian_means,
    gaussian_covariances,
    gaussian_sh_coefficients,
    gaussian_opacities,
    scale_invariant: bool = True,
):
    """Render one image per batch entry; returns (B, 3, H, W).

    extrinsics: (B, 4, 4) camera-to-world; intrinsics: (B, 3, 3) normalised by the image size; near / far: (B,);
    background_color: (B, 3); gaussian_means (B, G, 3); gaussian_covariances (B, G, 3, 3);
    gaussian_sh_coefficients (B, G, 3, n); gaussian_opacities (B, G).
    scale_invariant: rescale the scene by 1 / near before rasterising (numerical range).
    """
    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    # SH degree from the number of coefficients per colour channel: n = (degree + 1)^2, so n = 1 is degree 0
    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix
    tan_x, tan_y = tan_fov_x.tolist(), tan_fov_y.tolist()  # one host sync instead of two per view
    cov3d = cov_upper_triangle(gaussian_covariances)

    all_images = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_x[i],
            tanfovy=tan_y[i],
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        image, _ = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i],
            colors_precomp=None,
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=cov3d[i],
        )
        all_images.append(image)
    return torch.stack(all_images)
