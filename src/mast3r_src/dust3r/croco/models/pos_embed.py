# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# Modified for CSG-Fusion: RoPE2D is TorchRoPE2D (bit-identical, cached PyTorch RoPE); the curope kernel
# (cuRoPE2D) is opt-in.


# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------



import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [n_cls_token+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token>0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


#----------------------------------------------------------
# RoPE2D: RoPE implementation in 2D
#----------------------------------------------------------

class TorchRoPE2D(torch.nn.Module):
    """RoPE2D in plain PyTorch (used when the curope CUDA extension is not built).

    Every output element is ``t * cos + rotate_half(t) * sin``. The cos / sin rows of a position tensor are gathered
    once and reused by every attention layer (the encoder and decoder blocks all receive the same position tensors),
    and the y / x halves are rotated in one op, without a host sync.
    """

    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D,seq_len,device,dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos() # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D,seq_len,device,dtype] = (cos,sin)
        return self.cache[D,seq_len,device,dtype]

    def _recent(self, positions, key, make):
        # remembered for the last few position tensors: every block of the encoder / decoder gets the
        # same tensor, so this is computed once per forward instead of once per attention call
        recent = self.__dict__.setdefault('_recent_cache', [])
        for p, k, v in recent:
            if p is positions and k == key:
                return v
        v = make()
        recent.insert(0, (positions, key, v))
        del recent[8:]
        return v

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _gathered_cos_sin(self, tokens, positions):
        """(B, 1, N, 2*D) cos and sin rows for the [y | x] halves of each token."""
        D = tokens.size(3) // 2
        def make():
            # Table length: the token count bounds every grid position (y < h, x < w, N = h * w) and row i of
            # the table does not depend on its length, so this gives the same rows as int(positions.max()) + 1
            # without reading the positions back to the host.
            cos, sin = self.get_cos_sin(D, positions.shape[1], tokens.device, tokens.dtype)
            emb = torch.nn.functional.embedding
            return (torch.cat((emb(positions[:, :, 0], cos), emb(positions[:, :, 1], cos)), dim=-1)[:, None],
                    torch.cat((emb(positions[:, :, 0], sin), emb(positions[:, :, 1], sin)), dim=-1)[:, None])
        return self._recent(positions, ('cos_sin', D, tokens.device, tokens.dtype), make)

    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
        assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
        cos, sin = self._gathered_cos_sin(tokens, positions)
        # [y1 y2 x1 x2] -> [-y2 y1 -x2 x1]: rotate_half of each half, in one op
        t = tokens.unflatten(-1, (2, 2, tokens.size(3) // 4))
        rotated = torch.stack((-t[..., 1, :], t[..., 0, :]), dim=-2).flatten(-3)
        # contiguous output, like the torch.cat of the original (the attention matmuls see the same layout)
        out = torch.empty(tokens.shape, dtype=tokens.dtype, device=tokens.device)
        return torch.add(tokens * cos, rotated * sin, out=out)


# The PyTorch RoPE is the default: every CSG-Fusion number so far was produced with it (the curope
# extension did not build on PyTorch >= 2.x). The CUDA kernel (built with --use_fast_math, so not
# bit-identical) is opt-in, through the perf.rope switch in main.py, and None when it is not built.
try:
    from models.curope import cuRoPE2D
except ImportError:
    cuRoPE2D = None
RoPE2D = TorchRoPE2D
