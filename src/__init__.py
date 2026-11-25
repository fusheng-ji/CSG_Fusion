"""CSG-Fusion model code.

csg_gaussian_head      Gaussian head on the MASt3R backbone (filter + feature aggregation + Gaussian DPT heads)
fast_matching          Matching-based Gaussian Filter (Sec. 4.1) of the MASt3R backbone
matching_filter        the filter on SciPy matching matrices (VGGT backbone)
feature_aggregation    Mask-based Feature Aggregation (Sec. 4.2)
dpt_gs_head            Gaussian-parameter DPT heads
vggt_backbone          VGGT-1B backbone variant
backbone_graph         CUDA-graph replay of the frozen MASt3R backbone (performance only)
multiview              inference on 2-8 views with the two-view model
fusion_plugin          training-free fusion plug-in for other feed-forward methods
pixelsplat_src         renderer (adapted from pixelSplat)
mast3r_src             vendored MASt3R / DUSt3R
"""
