"""CSG-Fusion itself, under exactly the same protocol as the baselines (same samples, mask, metrics, timing).

The model is loaded with ``main.load_model`` (any variant; detected from the checkpoint) and run through its
own ``forward`` / ``_render``, i.e. the code path of ``main.CSGFusion.test_step``. Context views are
rendered by handing the decoder a batch whose targets are the context views.

With ``mv_mode`` set (CLI: --mv-mode / --mv-filter / --ref-fusion), or with more than two context views, the
Gaussians come from the multi-view inference of ``src/multiview.py`` instead.
"""
from .common import BaselineEvaluator


class CSGFusionEvaluator(BaselineEvaluator):
    name = 'csg_fusion'
    repo = None          # no upstream repo, no module sandbox
    mv_mode = None       # None: two-view forward (two context views); 'pairwise' | 'anchored' | 'joint': src/multiview.py
    mv_filter = 'all'
    ref_fusion = 'concat'

    def build(self, overrides):
        import main
        if overrides:
            raise ValueError('csg_fusion takes no overrides; the config is read from the checkpoint')
        self.model = main.load_model(str(self.checkpoint), self.device).eval()

    def prepare(self, batch):
        return batch

    def encode(self, inputs):
        if self.mv_mode is None and len(inputs['context']) == 2:
            view1, view2 = inputs['context']
            return self.model.forward(view1, view2)
        from src.multiview import predict_multiview
        preds, self.encode_stats = predict_multiview(self.model, inputs['context'], mode=self.mv_mode or 'pairwise',
                                                     filter=self.mv_filter, ref_fusion=self.ref_fusion)
        return preds

    def count_gaussians(self, out):
        pred1, pred2 = out                        # view 1 keeps all of its Gaussians, view 2 those in the filter's mask
        return pred1['means'][0, ..., 0].numel() + int(pred2['mask'][0].sum())

    def render(self, gaussians, inputs, views):
        if isinstance(gaussians, list):           # multi-view: [reference, view 2, ..., view N]
            from src.multiview import render_multiview
            return render_multiview(self.model, inputs, gaussians, views)
        pred1, pred2 = gaussians
        color, _ = self.model._render(dict(inputs, target=inputs[views]), pred1, pred2)
        return color
