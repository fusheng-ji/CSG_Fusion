"""Evaluation / loss mask (paper Sec. 4.3): target-view pixels that are visible in a context view.

A target pixel counts if its ground-truth depth is valid, it projects into the frustum of at least one context
view, and for at least one context view the ground-truth depth sampled at the projection agrees with the
point's depth in that view within ``atol = 0.1`` (scene units, i.e. 10 cm on metric ScanNet++ depth), i.e. a
context view sees the same surface rather than an occluder. (grid_sample pads with zeros, so a view whose
image the point misses does not match a valid depth.) Losses and metrics are computed on these pixels
(``loss.apply_mask`` / ``loss.average_over_mask``).
"""
import torch

from utils.geometry import unproject_depth, world_space_to_camera_space, camera_space_to_pixel_space

@torch.no_grad()
def calculate_in_frustum_mask(depth_1, intrinsics_1, c2w_1, depth_2, intrinsics_2, c2w_2):
    """
    A function that takes in the depth, intrinsics and c2w matrices of two sets
    of views, and then works out which of the pixels in the first set of views
    has a direct corresponding pixel in any of views in the second set

    Args:
        depth_1: (b, v1, h, w)
        intrinsics_1: (b, v1, 3, 3)
        c2w_1: (b, v1, 4, 4)
        depth_2: (b, v2, h, w)
        intrinsics_2: (b, v2, 3, 3)
        c2w_2: (b, v2, 4, 4)

    Returns:
        torch.Tensor: bool mask with shape (b, v1, h, w), True = the pixel is visible in a view of the
        second set (and has valid depth).
    """

    _, v1, h, w = depth_1.shape
    _, v2, _, _ = depth_2.shape

    # Unproject the depth to get the 3D points in world space
    points_3d = unproject_depth(depth_1[..., None], intrinsics_1, c2w_1)  # (b, v1, h, w, 3)

    # Project the 3D points into the pixel space of all the second views simultaneously
    camera_points = world_space_to_camera_space(points_3d, c2w_2)  # (b, v1, v2, h, w, 3)
    points_2d = camera_space_to_pixel_space(camera_points, intrinsics_2)  # (b, v1, v2, h, w, 2)

    # Calculate the depth of each point
    rendered_depth = camera_points[..., 2]  # (b, v1, v2, h, w)

    # Three conditions determine whether a point is kept in the mask

    # Condition 1: Check if the points are in the frustum of any of the v2 views
    in_frustum_mask = (
        (points_2d[..., 0] > 0) &
        (points_2d[..., 0] < w) &
        (points_2d[..., 1] > 0) &
        (points_2d[..., 1] < h)
    )  # (b, v1, v2, h, w)
    in_frustum_mask = in_frustum_mask.any(dim=-3)  # (b, v1, h, w)

    # Condition 2: Check if the points have non-zero (i.e. valid) depth in the input view
    non_zero_depth = depth_1 > 1e-6

    # Condition 3: Check if the points have matching depth to any of the v2
    # views torch.nn.functional.grid_sample expects the input coordinates to
    # be normalized to the range [-1, 1], so normalize first
    points_2d[..., 0] /= w
    points_2d[..., 1] /= h
    points_2d = points_2d * 2 - 1
    # Sample every (b, i, j) pair in one grid_sample call
    b, _, _, _ = depth_1.shape
    coords = points_2d.reshape(b * v1 * v2, h, w, 2)
    depth = depth_2[:, None].expand(b, v1, v2, h, w).reshape(b * v1 * v2, 1, h, w)
    sampled_depths = torch.nn.functional.grid_sample(depth, coords, align_corners=False)[:, 0].view(b, v1, v2, h, w)
    # same surface: depths agree within 0.1 (10 cm on metric data), plus isclose's default rtol of 1e-5
    matching_depth = torch.isclose(rendered_depth, sampled_depths, atol=1e-1)

    matching_depth = matching_depth.any(dim=-3)  # (..., v1, h, w)

    return in_frustum_mask & non_zero_depth & matching_depth


@torch.no_grad()
def calculate_loss_mask(batch):
    '''Calculate the loss mask (b, v_target, h, w) for the target views in the batch.'''

    target_depth = torch.stack([target_view['depthmap'] for target_view in batch['target']], dim=1)
    target_intrinsics = torch.stack([target_view['camera_intrinsics'] for target_view in batch['target']], dim=1)
    target_c2w = torch.stack([target_view['camera_pose'] for target_view in batch['target']], dim=1)
    context_depth = torch.stack([context_view['depthmap'] for context_view in batch['context']], dim=1)
    context_intrinsics = torch.stack([context_view['camera_intrinsics'] for context_view in batch['context']], dim=1)
    context_c2w = torch.stack([context_view['camera_pose'] for context_view in batch['context']], dim=1)

    target_intrinsics = target_intrinsics[..., :3, :3]
    context_intrinsics = context_intrinsics[..., :3, :3]

    mask = calculate_in_frustum_mask(
        target_depth, target_intrinsics, target_c2w,
        context_depth, context_intrinsics, context_c2w
    )
    return mask
