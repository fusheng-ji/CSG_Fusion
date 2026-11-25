"""Mask-based Feature Aggregation (paper Sec. 4.2).

``FeatureFusion`` updates the patch tokens of one view (the source / query view, view 1 by default) with
cross-attention to the patch tokens of the other view (target / key view), restricted by the patch matching
matrix of the Matching-based Gaussian Filter (Sec. 4.1): a query patch only attends to the key patches that
the matching matrix allows. Used by both backbones (MASt3R: 32 x 32 tokens of 3328 channels; VGGT: 37 x 37
tokens of 4 x 2048 channels).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """
    Feature fusion module for features already split into patches.
    Input shape: [B, N, C] with N = grid_size ** 2 patch tokens in row-major order
    (32 x 32 = 1024 for MASt3R at 512 x 512, 37 x 37 = 1369 for VGGT at 518 x 518).

    Data flow: project both views to ``hidden_dim``; masked cross-attention (queries: source patches,
    keys / values: target patches) and an MLP, each with a residual; a per-patch sigmoid gate blends the
    projected source features with the attended ones; project back to ``output_dim`` and add the input
    tokens (outer residual).
    """
    def __init__(
        self,
        input_dim=768,          # per-patch feature dim
        hidden_dim=512,         # hidden dim
        output_dim=768,         # output dim, same as input
        num_heads=8,            # attention heads
        dropout=0.1,            # dropout rate
        neighborhood_size=0,    # neighborhood size
        grid_size=32,           # grid size (default 32x32=1024 patches)
    ):
        super().__init__()
        # Training fixes (main.py model.*), off in the published model:
        self.mask_transpose_fix = False  # rows of the matching matrix are view-1 patches = the queries
        self.skip_unmatched = False      # queries without any match get no attention update

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.neighborhood_size = neighborhood_size
        self.grid_size = grid_size

        # Down-projection to the hidden dim
        self.src_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.tgt_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Attention layer
        self.cross_attention = CrossAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Up-projection back to the output dim
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU()
        )

        # Fusion gate: one weight in (0, 1) per patch, from the concatenated original and attended features
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def expand_neighborhood(self, matching_matrix):
        """Dilate the (B, N, N) matching matrix over the query patches: query patch (i, j) takes the
        element-wise max of the rows of its (2n+1)^2 spatial neighbours, so it may attend to every key patch
        matched by a neighbouring query. The key dimension is not dilated."""
        if self.neighborhood_size == 0:
            return matching_matrix

        B, N, _ = matching_matrix.shape

        # Grid size
        grid_size = self.grid_size
        assert N == grid_size * grid_size, f"Patch count {N} and grid size {grid_size}*{grid_size} do not match"

        # Reshape the matching matrix to a spatial layout
        spatial_match = matching_matrix.view(B, grid_size, grid_size, grid_size, grid_size)
        expanded_match = torch.zeros_like(spatial_match)

        # Expand the neighbourhood: each query patch (i, j) takes the max over the (2n+1)^2 neighbouring
        # query patches. Same result as the per-cell loop, one slice op per offset.
        G, n = grid_size, self.neighborhood_size
        for di in range(-n, n + 1):
            for dj in range(-n, n + 1):
                i0, i1 = max(0, -di), min(G, G - di)
                j0, j1 = max(0, -dj), min(G, G - dj)
                expanded_match[:, i0:i1, j0:j1] = torch.max(
                    expanded_match[:, i0:i1, j0:j1],
                    spatial_match[:, i0 + di:i1 + di, j0 + dj:j1 + dj]
                )

        # Back to the original shape
        return expanded_match.view(B, N, N)

    def forward(self, src_patches, tgt_patches, matching_matrix):
        """
        Fuse patch features.

        Args:
            src_patches: source (query) patches [B, N, C]
            tgt_patches: target (key / value) patches [B, N, C]
            matching_matrix: patch matching matrix, rows = source patches, cols = target patches: a dense (B, N, N)
                tensor of match counts, or a list of B SciPy CSR matrices

        Returns:
            fused features [B, N, C]
        """
        B, N, _ = src_patches.shape
        attention_matrix = self._attention_matrix(matching_matrix, B, N, src_patches.device)
        return self._fuse(src_patches, tgt_patches, attention_matrix)

    def forward_multi(self, src_patches, tgt_patches_list, matching_matrices):
        """Multi-view inference (src/multiview.py): the source patches attend to the patches of several target
        views at once. Each view's matching matrix is handled as in ``forward``; the views' keys / values and
        attention masks are concatenated along the key axis, so a query attends to its matched patches in
        every view (one softmax over all of them). With one target view this is ``forward``."""
        B, N, _ = src_patches.shape
        attention_matrix = torch.cat([self._attention_matrix(m, B, N, src_patches.device) for m in matching_matrices],
                                     dim=2)
        return self._fuse(src_patches, torch.cat(tgt_patches_list, dim=1), attention_matrix)

    def _attention_matrix(self, matching_matrix_in, B, N, device):
        """Matching matrix (any of the forms ``forward`` takes) -> the (B, N queries, N keys) float matrix whose
        zeros the attention masks out: densified, transposed (unless mask_transpose_fix) and dilated."""
        if torch.is_tensor(matching_matrix_in):
            # dense (B, N, N) patch counts (src/fast_matching.patch_counts)
            matching_matrix = matching_matrix_in.to(device=device, dtype=torch.float32)
        else:  # a list of B CSR matrices (VGGT backbone)
            matching_matrix = torch.zeros(B, N, N, device=device)
            for i, mat in enumerate(matching_matrix_in):
                matching_matrix[i] = torch.from_numpy(mat.toarray()).to(device)
        # Check the matching matrix shape
        assert matching_matrix.size(1) == N and matching_matrix.size(2) == N, \
            f"Matching matrix shape {matching_matrix.shape} and feature size {N}x{N} do not match"

        # The rows of the matching matrix are already the source (query) patches. The default transpose
        # (used by the published model) makes query patch i attend to the target patches
        # j for which source patch j matched target patch i, i.e. query i uses the match list of target
        # patch i. mask_transpose_fix uses the matrix as is: query i attends to the target patches it matched.
        transposed_matching = matching_matrix if self.mask_transpose_fix else matching_matrix.transpose(1, 2)

        # Expand the matching matrix with neighbors
        if self.neighborhood_size > 0:
            return self.expand_neighborhood(transposed_matching)
        return transposed_matching

    def _fuse(self, src_patches, tgt_patches, expanded_matching):
        """Masked cross-attention + MLP + gate + outer residual; ``expanded_matching``: (B, Nq, Nk), 0 = masked."""
        # Down-project
        src_features = self.src_proj(src_patches)  # [B, N, hidden_dim]
        tgt_features = self.tgt_proj(tgt_patches)  # [B, N, hidden_dim]

        # Additive attention mask: -1e9 on unmatched (query, key) pairs restricts each query to its matched
        # key patches. For a query with no match at all every logit becomes -1e9 (float32 cannot resolve the
        # original logits at that magnitude), so its softmax is a uniform average over all key patches
        # (skip_unmatched zeroes that update).
        # float32 -1e9 / 0 (Python scalars: no host-to-device copies)
        attention_mask = torch.where(expanded_matching == 0, -1e9, 0.0).to(torch.float32)

        # Keep the source features for adaptive fusion
        src_original = src_features.clone()  # clone() to avoid in-place ops

        # Cross-attention: source as query, target as key/value
        src_attended = self.cross_attention(
            q=self.norm1(src_features),
            k=tgt_features,
            v=tgt_features,
            mask=attention_mask
        )
        if self.skip_unmatched:  # all-masked rows would softmax to a uniform average over every patch
            src_attended = src_attended * (expanded_matching != 0).any(dim=-1, keepdim=True).to(src_attended.dtype)

        # Residual
        src_features = src_features + src_attended

        # MLP
        src_mlp_output = self.mlp(self.norm2(src_features))
        src_features = src_features + src_mlp_output  # avoid in-place ops

        # Adaptive fusion: the per-patch gate blends the projected source features (gate -> 0) with the
        # attended ones (gate -> 1), so a patch can fall back to its own features
        combined = torch.cat([src_original, src_features], dim=-1)
        gate = self.fusion_gate(combined)

        # Weighted blend
        fused_features = src_original * (1 - gate) + src_features * gate

        # Up-project back to the output dim
        output = self.output_proj(fused_features)

        # Outer residual: the input tokens are kept and the module adds an update on top of them
        output = output + src_patches

        return output


class CrossAttention(nn.Module):
    """Multi-head cross-attention with an optional additive mask ([B, Nq, Nk] or [B, heads, Nq, Nk])."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Nq, C = q.shape
        Nk = k.shape[1]

        # Projections
        q = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply mask
        if mask is not None:
            # Make sure the mask has the right shape
            if mask.dim() == 3:  # [B, Nq, Nk]
                mask = mask.unsqueeze(1)  # [B, 1, Nq, Nk]
            attn = attn + mask

        # Softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        out = self.out_proj(out)
        out = self.out_drop(out)

        return out
