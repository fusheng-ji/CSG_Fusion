"""Geometry helpers: intrinsics normalisation, quaternion -> rotation, Gaussian covariances (R S S^T R^T)
and the camera / world / pixel transforms of the loss mask (``utils/loss_mask.py``).

The covariance products have performance-only variants (``perf.elementwise_covariance`` and the exact
inference path ``EXACT_DIAG_MM``); see ``build_covariance``.
"""
import einops
import torch

# --- Intrinsics Transformations ---

def normalize_intrinsics(intrinsics, image_shape):
    '''Normalize an intrinsics matrix given the image shape (H, W): row 0 / W, row 1 / H.'''
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, :] /= image_shape[1]
    intrinsics[..., 1, :] /= image_shape[0]
    return intrinsics


# --- Quaternions, Rotations and Scales ---

def quaternion_to_matrix(quaternions, eps: float = 1e-8):
    '''
    Convert the 4-dimensional quaternions to 3x3 rotation matrices.
    This is adapted from Pytorch3D:
    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    '''

    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return einops.rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def _round_tf32(x):
    '''Round fp32 to the TF32 grid (10 mantissa bits, round-to-nearest-even), as cuBLAS does
    with its operands when torch.backends.cuda.matmul.allow_tf32 is on.'''
    i = x.contiguous().view(torch.int32)
    i = i + 0xFFF + ((i >> 13) & 1)
    return (i & ~0x1FFF).view(torch.float32)


def _mm3(a, b, tf32):
    if tf32 and a.is_cuda and a.dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32:
        a, b = _round_tf32(a), _round_tf32(b)
    # c_ij = (a_i0 b_0j + a_i1 b_1j) + a_i2 b_2j
    return (a[..., :, 0, None] * b[..., None, 0, :] + a[..., :, 1, None] * b[..., None, 1, :]) + a[..., :, 2, None] * b[..., None, 2, :]


class _MatMul3x3(torch.autograd.Function):
    '''Batched 3x3 a @ b with element-wise kernels.

    torch.matmul on millions of 3x3 matrices launches GEMM tiles that are mostly padding. This computes
    the same products as the bmm calls autograd makes:
    forward a @ b and grad_a = grad @ b^T in TF32 when allow_tf32 is on (operands rounded to
    nearest, like cuBLAS), grad_b = a^T @ grad in fp32 (cuBLAS takes its non-TF32 kernel for a
    transposed first operand). Results agree with the bmm path to an ulp or so, but not bit for bit
    (tensor-core accumulation, and cuBLAS computes some batch remainders in plain fp32), and the
    rasterizer gradients amplify ulp-level covariance changes, so it is an opt-in.'''

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _mm3(a, b, tf32=True)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        grad_a = _mm3(grad, b.transpose(-1, -2), tf32=True) if ctx.needs_input_grad[0] else None
        grad_b = _mm3(a.transpose(-1, -2), grad, tf32=False) if ctx.needs_input_grad[1] else None
        return grad_a, grad_b


def matmul3x3(a, b, elementwise=True):
    if not elementwise:
        return a @ b
    a, b = torch.broadcast_tensors(a, b)
    return _MatMul3x3.apply(a, b)


_DIAG_MM_TF32 = {}   # (shapes, strides, dtype, device, allow_tf32) -> per-matrix bool: cuBLAS used its TF32 kernel


def _diag_mm_modes(a, d):
    """Which matrices of the batched product ``a @ d`` (d diagonal) cuBLAS computes with its TF32 kernel.

    PyTorch hands cuBLAS the batch in chunks of at most 65535 matrices, and cuBLAS picks a kernel per chunk
    from its shape: the TF32 tensor-core kernel (operands rounded to TF32) or the fp32 SIMT kernel (the
    trailing chunk of 4 matrices at 512x512, all of a small batch). That depends only on shapes, strides,
    dtype and allow_tf32, so it is probed once per configuration, with operands of the same layout whose
    TF32 and fp32 products differ in every element.
    """
    tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    key = (tuple(a.shape), a.stride(), tuple(d.shape), d.stride(), a.dtype, a.device, tf32)
    if key not in _DIAG_MM_TF32:
        va, vd = 1 + 2 ** -11 + 2 ** -20, 1 + 2 ** -11 + 2 ** -21     # TF32 (round to nearest): 1 + 2^-10
        pa = torch.empty_strided(a.shape, a.stride(), dtype=a.dtype, device=a.device).fill_(va)
        pd = torch.empty_strided(d.shape, d.stride(), dtype=d.dtype, device=d.device).zero_()
        pd.diagonal(dim1=-2, dim2=-1).fill_(vd)
        out = pa @ pd
        diag = pd.diagonal(dim1=-2, dim2=-1)[..., None, :]
        as_tf32 = out == _round_tf32(pa) * _round_tf32(diag)
        as_fp32 = out == pa * diag
        mode = as_tf32.flatten(-2).all(-1)
        if not bool((as_tf32 | as_fp32).all()) or not bool((as_tf32.flatten(-2).any(-1) <= mode).all()):
            mode = None          # neither product, or mixed within a matrix: keep using cuBLAS
        _DIAG_MM_TF32[key] = mode
    return _DIAG_MM_TF32[key]


def _diag_mm_exact(a, d, diag):
    """``a @ d`` for a diagonal ``d`` (``diag``: its diagonal, (..., 3)), bit-identical to the cuBLAS result.

    Each output element has a single non-zero term a_ij * d_jj; adding exact zeros changes nothing, so cuBLAS
    returns that product of its operands as its kernel sees them: TF32-rounded or fp32 (a TF32 x TF32 product
    has at most 22 significant bits and is exact in fp32). As in cuBLAS, a non-finite a_ik (k != j) still turns
    c_ij into NaN through its 0 * inf term. Subnormal operands are the one case not covered (tensor cores may
    flush them).
    """
    mode = _diag_mm_modes(a, d)
    if mode is None:
        return a @ d
    diag = diag[..., None, :]
    out = torch.where(mode[..., None, None], _round_tf32(a) * _round_tf32(diag), a * diag)
    bad = ~torch.isfinite(a)
    others_bad = (bad.sum(-1, keepdim=True) - bad.int()) > 0          # some a_ik, k != j, is inf / NaN
    return torch.where(others_bad, float('nan'), out)


def build_covariance(scale, rotation_xyzw, elementwise=False):
    '''Build the 3x3 covariance matrix from the three dimensional scale and the
    four dimension quaternion. elementwise=True uses matmul3x3 (faster, agrees to ~1 ulp).

    Without gradients (inference), the two products with the diagonal scale matrix are computed
    element-wise with exactly cuBLAS's rounding (_diag_mm_exact), bit-identical to the bmm path and
    faster; the last product (R S S^T) R^T stays on cuBLAS.'''
    scale_diag = scale
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    if not elementwise and scale.is_cuda and scale.dtype == torch.float32 and not (
            torch.is_grad_enabled() and (scale.requires_grad or rotation.requires_grad)):
        scale_t = einops.rearrange(scale, "... i j -> ... j i")
        rs = _diag_mm_exact(rotation, scale, scale_diag)
        rss = _diag_mm_exact(rs, scale_t, scale_diag)
        return rss @ einops.rearrange(rotation, "... i j -> ... j i")
    mm = lambda a, b: matmul3x3(a, b, elementwise)
    return mm(mm(mm(
        rotation,
        scale),
        einops.rearrange(scale, "... i j -> ... j i")),
        einops.rearrange(rotation, "... i j -> ... j i"))


# --- Projections ---

def homogenize_points(points):
    """Append a '1' along the final dimension of the tensor (i.e. convert xyz->xyz1)"""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def normalize_homogenous_points(points):
    """Normalize the point vectors"""
    return points / points[..., -1:]


def pixel_space_to_camera_space(pixel_space_points, depth, intrinsics):
    """
    Convert pixel space points to camera space points.

    Args:
        pixel_space_points (torch.Tensor): Pixel space points with shape (h, w, 2)
        depth (torch.Tensor): Depth map with shape (b, v, h, w, 1)
        intrinsics (torch.Tensor): Camera intrinsics with shape (b, v, 3, 3)

    Returns:
        torch.Tensor: Camera space points with shape (b, v, h, w, 3).
    """
    pixel_space_points = homogenize_points(pixel_space_points)
    camera_space_points = torch.einsum('b v i j , h w j -> b v h w i', intrinsics.inverse(), pixel_space_points)
    camera_space_points = camera_space_points * depth
    return camera_space_points


def camera_space_to_world_space(camera_space_points, c2w):
    """
    Convert camera space points to world space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """
    camera_space_points = homogenize_points(camera_space_points)
    world_space_points = torch.einsum('b v i j , b v h w j -> b v h w i', c2w, camera_space_points)
    return world_space_points[..., :3]


def camera_space_to_pixel_space(camera_space_points, intrinsics):
    """
    Convert camera space points to pixel space points (perspective divide, then the intrinsics).

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v1, v2, h, w, 3)
        intrinsics (torch.Tensor): Pixel intrinsics of the second set of views with shape (b, v2, 3, 3)

    Returns:
        torch.Tensor: Pixel coordinates (x, y) with shape (b, v1, v2, h, w, 2).
    """
    camera_space_points = normalize_homogenous_points(camera_space_points)
    pixel_space_points = torch.einsum('b u i j , b v u h w j -> b v u h w i', intrinsics, camera_space_points)
    return pixel_space_points[..., :2]


def world_space_to_camera_space(world_space_points, c2w):
    """
    Convert world space points to the camera space of every view of a second set.

    Args:
        world_space_points (torch.Tensor): World space points with shape (b, v1, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 4, 4)

    Returns:
        torch.Tensor: Camera space points with shape (b, v1, v2, h, w, 3).
    """
    world_space_points = homogenize_points(world_space_points)
    camera_space_points = torch.einsum('b u i j , b v h w j -> b v u h w i', c2w.inverse(), world_space_points)
    return camera_space_points[..., :3]


def unproject_depth(depth, intrinsics, c2w):
    """
    Turn the depth map into a 3D point cloud in world space

    Args:
        depth: (b, v, h, w, 1)
        intrinsics: (b, v, 3, 3)
        c2w: (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """

    # Compute indices of pixels
    h, w = depth.shape[-3], depth.shape[-2]
    x_grid, y_grid = torch.meshgrid(
        torch.arange(w, device=depth.device, dtype=torch.float32),
        torch.arange(h, device=depth.device, dtype=torch.float32),
        indexing='xy'
    )  # (h, w), (h, w)

    # Compute coordinates of pixels in camera space
    pixel_space_points = torch.stack((x_grid, y_grid), dim=-1)  # (..., h, w, 2)
    camera_points = pixel_space_to_camera_space(pixel_space_points, depth, intrinsics)  # (..., h, w, 3)

    # Convert points to world space
    world_points = camera_space_to_world_space(camera_points, c2w)  # (..., h, w, 3)

    return world_points
