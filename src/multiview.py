"""Multi-view inference: 3D Gaussians from N >= 2 unposed views with a trained two-view CSG-Fusion model.

No retraining. View 0 is the reference: its camera frame is the world frame (as view 1's is for two views)
and all of its Gaussians are kept. Every other view k contributes the Gaussians the Matching-based Gaussian
Filter keeps.

Three ways to get the per-view Gaussians:

    mode='pairwise' (both backbones)   the two-view model on pairs chained over a spanning tree. Every pair of
                                       views is matched (reciprocal nearest neighbours of the pair's matching
                                       descriptors); starting from the reference, each view is attached to the
                                       already placed view it shares the most matches with (its parent). The
                                       pair (parent, k) gives view k's Gaussians in the parent's frame, filtered
                                       against the parent; a similarity transform fitted on the parent's points
                                       (predicted in both frames) brings them into the reference frame. Views
                                       that do not overlap the reference are thus placed through views that do.
    mode='anchored' (both backbones)   every view k paired with the reference only, pairs (0, k): simpler, but a
                                       view that does not overlap the reference gets meaningless geometry.
    mode='joint'    (VGGT only)        one VGGT aggregator pass over all N views (one shared frame and metric
                                       scale), then src/vggt_backbone.VGGTGaussianHead.forward_multi:
                                       head 0 for the reference, head 1 for every other view, feature
                                       aggregation into the reference from all other views
                                       (ref_fusion='concat') or from the best-matching one ('best').

The reference Gaussians (pairwise / anchored) come from the reference's pair with the most matches.

and three filters:

    filter='none'   keep everything: the plain union of all views' Gaussians (the duplication baseline)
    filter='ref'    each view filtered only against the view it was predicted with (pairwise: its parent;
                    anchored / joint: the reference), as the two-view model does
    filter='all'    additionally, each view is filtered against every view placed before it: the same matching
                    and the same 3D distance test, on the reference-frame points of both views. The descriptors
                    come from a pass on that pair (pairwise / anchored) or from the joint pass.

``render_multiview`` renders the result with the model's decoder (``forward_multi``).
"""
import contextlib

import einops
import numpy as np
import omegaconf
import torch

import src.fast_matching as fast_matching
import utils.sh_utils as sh_utils

MODES = ('pairwise', 'anchored', 'joint')
FILTERS = ('none', 'ref', 'all')
REF_FUSIONS = ('concat', 'best')


@contextlib.contextmanager
def _filter_disabled(model):
    """A negative distance threshold makes no view-2 Gaussian redundant (the matches, and so the feature
    aggregation, are unchanged)."""
    head = model.gaussian_head
    old = head.dist_threshold
    head.dist_threshold = -1.0
    try:
        yield
    finally:
        head.dist_threshold = old


def _pair_descriptors(model, view_a, view_b):
    """Matching descriptors of views a and b, predicted on the pair (a, b): (B, H, W, D) each."""
    if model.backbone == 'vggt':
        res_a, res_b, *_ = model.encoder(view_a, view_b, view_a['original_img'].shape[-2:])
        return res_a['desc'], res_b['desc']
    enc = model.encoder
    bf16 = bool(omegaconf.OmegaConf.select(model.config, 'perf.backbone_bf16', default=False))
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):      # as in main.CSGFusion.forward
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = enc._encode_symmetrized(view_a, view_b)
        dec1, dec2 = enc._decoder(feat1, pos1, feat2, pos2)
    res1 = enc._downstream_head(1, [tok.float() for tok in dec1], shape1)
    res2 = enc._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1['desc'], res2['desc']


def _add_rgb(model, view, pred):
    """Per-view RGB residual (main.CSGFusion._add_rgb_residual with learn_residual, one view at a time)."""
    config = model.config
    if not hasattr(config, 'learn_residual') or config.learn_residual:
        rgb = torch.zeros_like(pred['sh'])
        rgb[..., 0] = sh_utils.RGB2SH(einops.rearrange(view['original_img'], 'b c h w -> b h w c'))
        pred['sh'] = pred['sh'] + rgb


def _matches(model, views, cache, a, b):
    """Matches of views a and b (batch item 0; B = 1), from the descriptors of a pass on the pair; cached."""
    key = (min(a, b), max(a, b))
    if key not in cache:
        desc_i, desc_j = _pair_descriptors(model, views[key[0]], views[key[1]])
        cache[key] = fast_matching.reciprocal_matches(desc_i[0].detach().float(), desc_j[0].detach().float(),
                                                      subsample=getattr(model.gaussian_head, 'matching_subsample', 8))
    return cache[key]


def _counts(model, views, cache, a, b):
    """Patch match counts of views a and b, rows = view a, columns = view b."""
    m1, m2, valid = _matches(model, views, cache, a, b)
    return fast_matching.patch_counts(m1, m2, valid) if a < b else fast_matching.patch_counts(m2, m1, valid)


def _spanning_tree(model, views, cache):
    """Prim's maximum spanning tree from view 0 on the numbers of matches: [(parent, child)] in placement order."""
    n = len(views)
    weight = {(i, j): int(_matches(model, views, cache, i, j)[2].sum()) for i in range(n) for j in range(i + 1, n)}
    placed, edges = [0], []
    while len(placed) < n:
        p, k = max(((p, k) for p in placed for k in range(n) if k not in placed),
                   key=lambda e: weight[(min(e), max(e))])
        edges.append((p, k))
        placed.append(k)
    return edges


def _similarity(src, dst, weight):
    """Umeyama: s, R, t minimising sum weight * |s R src + t - dst|^2; src, dst: (n, 3)."""
    src, dst, w = src.double(), dst.double(), weight.double()[:, None] / weight.sum()
    mu_s, mu_d = (w * src).sum(0), (w * dst).sum(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = (w * xd).T @ xs
    U, S, Vt = torch.linalg.svd(cov)
    D = torch.eye(3, dtype=cov.dtype, device=cov.device)
    D[2, 2] = torch.sign(torch.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * D.diagonal()).sum() / (w * xs.pow(2)).sum()
    return s, R, mu_d - s * R @ mu_s


def _to_reference(pred, pts_src, pts_dst, conf):
    """Move a view-k prediction from its parent's frame to the reference frame with the similarity that maps the
    parent's points predicted in the parent's frame (pts_src) onto the same points in the reference frame (pts_dst),
    fitted on the more confident half of the pixels (every 4th)."""
    src, dst, c = pts_src.reshape(-1, 3)[::4], pts_dst.reshape(-1, 3)[::4], conf.reshape(-1)[::4]
    keep = c >= c.median()
    s, R, t = _similarity(src[keep], dst[keep], torch.ones_like(c[keep]))
    s, R, t = s.float(), R.float(), t.float()
    for key in ('means_in_other_view', 'pts3d_in_other_view'):
        pred[key] = s * pred[key] @ R.T + t
    pred['covariances'] = (s * s) * R @ pred['covariances'] @ R.T
    if pred['sh'].shape[-1] != 1:
        raise NotImplementedError('rotating higher-order SH is not implemented (the published models use degree 0)')


@torch.no_grad()
def predict_multiview(model, views, mode='pairwise', filter='all', ref_fusion='concat'):
    """Gaussians of N views in the frame of ``views[0]``.

    Returns ``(preds, stats)``. ``preds[0]`` is the reference view (keys of a two-view ``pred1``);
    ``preds[1:]`` (in input order) have the keys of a two-view ``pred2`` (``means_in_other_view``,
    ``pts3d_in_other_view``, ``mask`` with 1 = keep). ``stats``: Gaussian counts. Render with ``render_multiview``."""
    if mode not in MODES or filter not in FILTERS or ref_fusion not in REF_FUSIONS:
        raise ValueError(f'mode {mode!r} / filter {filter!r} / ref_fusion {ref_fusion!r}')
    n = len(views)
    if n < 2:
        raise ValueError('multi-view inference needs at least two views')
    head = model.gaussian_head
    cache = {}
    with _filter_disabled(model) if filter == 'none' else contextlib.nullcontext():
        if mode == 'pairwise':
            if views[0]['original_img'].shape[0] != 1:
                raise ValueError("mode='pairwise' takes one scene at a time (batch size 1)")
            edges = _spanning_tree(model, views, cache)
            order = [0] + [k for _, k in edges]
            preds = [None] * n
            ref_candidates = []
            for p, k in edges:
                pred_p, pred_k = model.forward(views[p], views[k])
                if p == 0:
                    ref_candidates.append((int(cache[(0, k)][2].sum()), pred_p))
                else:   # the parent is already in the reference frame (edges are in placement order)
                    _to_reference(pred_k, pred_p['pts3d'], preds[p]['pts3d_in_other_view'], pred_p['conf'])
                preds[k] = pred_k
            preds[0] = max(ref_candidates, key=lambda c: c[0])[1]
        elif mode == 'anchored':
            if filter == 'all' and views[0]['original_img'].shape[0] != 1:
                raise ValueError("filter='all' with mode='anchored' takes one scene at a time (batch size 1)")
            pairs = [model.forward(views[0], view) for view in views[1:]]
            n_matches = [sum(float(cm.sum()) for cm in p2['cross_mat']) for _, p2 in pairs]
            preds = [pairs[int(np.argmax(n_matches))][0]] + [p2 for _, p2 in pairs]
            order = list(range(n))
        else:
            if model.backbone != 'vggt':
                raise ValueError("mode='joint' needs the VGGT backbone (MASt3R is a pairwise network)")
            res, tokens, patch_start, imgs = model.encoder.forward_views(views, views[0]['original_img'].shape[-2:])
            preds = head.forward_multi(res, tokens, patch_start, imgs, ref_fusion=ref_fusion)
            for view, pred in zip(views, preds):
                pred['covariances'] = model._covariance(pred['scales'], pred['rotations'])
                _add_rgb(model, view, pred)
            for pred in preds[1:]:
                pred['pts3d_in_other_view'] = pred.pop('pts3d')
                pred['means_in_other_view'] = pred.pop('means')
            order = list(range(n))

    if filter == 'all':
        parent = {k: p for p, k in edges} if mode == 'pairwise' else {k: 0 for k in range(1, n)}
        points = lambda v: preds[v]['pts3d'] if v == 0 else preds[v]['pts3d_in_other_view']
        for pos, k in enumerate(order[1:], start=1):
            red = torch.zeros_like(preds[k]['mask'][..., 0], dtype=torch.bool)
            for j in order[:pos]:
                if j == parent[k]:
                    continue                        # already filtered against it by the pair / joint head
                if mode == 'joint':
                    for i in range(red.shape[0]):
                        m1, m2, valid = fast_matching.reciprocal_matches(preds[j]['desc'][i].detach().float(),
                                                                         preds[k]['desc'][i].detach().float())
                        red[i] |= fast_matching.distance_mask(points(j)[i].detach(), points(k)[i].detach(),
                                                              fast_matching.patch_counts(m1, m2, valid),
                                                              threshold=head.dist_threshold,
                                                              union=head.filter_union_matches)
                else:
                    red[0] |= fast_matching.distance_mask(points(j)[0].detach(), points(k)[0].detach(),
                                                          _counts(model, views, cache, j, k), threshold=head.dist_threshold,
                                                          union=head.filter_union_matches)
            preds[k]['mask'] = preds[k]['mask'] * (~red)[..., None].float()
            preds[k]['opacities'] = torch.where(preds[k]['mask'] == 0, 0.0, preds[k]['opacities'])

    others = preds[1:]
    hw = preds[0]['means'].shape[1] * preds[0]['means'].shape[2]
    kept = [float(p['mask'].sum()) / p['mask'].shape[0] for p in others]
    stats = {'num_gaussians': float(hw + sum(kept)), 'num_gaussians_before': float(hw * n),
             'view2_dropped': float(1 - sum(kept) / (hw * len(others)))}
    return preds, stats


def render_multiview(model, batch, preds, views='target'):
    """Render ``batch[views]`` from the multi-view Gaussians: (B, V, 3, H, W) in [0, 1]."""
    color, _ = model.decoder.forward_multi(dict(batch, target=batch[views]), preds, render_context=False)
    return color
