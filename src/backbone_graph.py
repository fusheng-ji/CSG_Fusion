"""CUDA-graph replay of the frozen MASt3R encoder blocks and decoder at inference (``perf.cuda_graph``).

A performance-only helper: it changes how kernels are launched, not what they compute.

The ViT-L encoder (24 blocks) and the two 12-block decoders launch thousands of small kernels per sample;
when the CPU cannot issue them as fast as the GPU runs them, inference is launch-bound.
A CUDA graph records the kernels once per input shape and replays them with one launch. The replayed
kernels, their order and their inputs are the same as in eager mode, so the outputs are bit-identical.

Only used without autograd (``torch.no_grad`` / inference); anything else runs eagerly. The patch embedding
stays eager (its boolean indexing reads back to the host, which a graph cannot do). Outputs are copied out
of the graph's static buffers, so results of earlier calls are never overwritten by later ones.
"""
import torch


class GraphedFunction:
    """Calls ``fn(*tensors)`` through a CUDA graph captured per input signature (shape / dtype / stride)."""

    def __init__(self, fn, before_capture=None, active=None, parameters=()):
        self.fn = fn
        self.parameters = list(parameters)     # weights used by fn: a graph replays their addresses at capture time
        self.before_capture = before_capture   # e.g. clear caches that must be recomputed inside the graph
        self.active = active                   # callable: False -> run eagerly (e.g. while training)
        self.enabled = True
        self.graphs = {}
        self.addresses = None

    def clear(self):
        """Drop the captured graphs. Needed after anything that moves the weights or frees the cuBLAS workspaces, such
        as the teardown of a Lightning Trainer (main.CSGFusion.teardown)."""
        self.graphs = {}
        self.addresses = None

    def __call__(self, *args):
        if (not self.enabled or (self.active is not None and not self.active()) or torch.is_grad_enabled() or not all(a.is_cuda for a in args)
                or torch.cuda.is_current_stream_capturing()):
            return self.fn(*args)
        key = tuple((a.shape, a.dtype, a.stride(), a.device) for a in args) + (
            torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype(), torch.is_inference_mode_enabled())
        addresses = [p.data_ptr() for p in self.parameters]
        if addresses != self.addresses:                    # first call, or the weights were moved (model.cpu() / .to())
            self.graphs, self.addresses = {}, addresses
        if key not in self.graphs:
            self.graphs[key] = self._capture(args)
        graph, static_in, static_out = self.graphs[key]
        for s, a in zip(static_in, args):
            s.copy_(a)
        graph.replay()
        return _clone(static_out)

    def _capture(self, args):
        static_in = [a.clone() for a in args]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):                    # warm-up: lazy init, cuBLAS handles, caches
            for _ in range(2):
                self.fn(*static_in)
        torch.cuda.current_stream().wait_stream(stream)
        if self.before_capture is not None:
            self.before_capture()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self.fn(*static_in)
        if self.before_capture is not None:
            self.before_capture()
        return graph, static_in, static_out


def _clone(x):
    if torch.is_tensor(x):
        return x.clone()
    return type(x)(_clone(v) for v in x)


def install(encoder, active=None):
    """Route ``encoder._encode_image`` (after the patch embedding) and ``encoder._decoder`` through graphs,
    while ``active()`` is true. Returns the two GraphedFunction objects (``.enabled`` switches them off)."""

    def clear_rope_caches():
        # the RoPE caches cos / sin rows per position tensor: inside the graph they must be computed from the
        # static input positions, not taken from the warm-up
        for m in encoder.modules():
            rope = getattr(m, 'rope', None)
            if rope is not None and '_recent_cache' in rope.__dict__:
                rope.__dict__['_recent_cache'].clear()

    def blocks(x, pos):
        for blk in encoder.enc_blocks:
            x = blk(x, pos)
        return (encoder.enc_norm(x),)

    def decoder(f1, pos1, f2, pos2):
        return tuple(tuple(t) for t in eager_decoder(f1, pos1, f2, pos2))

    eager_decoder = encoder._decoder
    weights = list(encoder.parameters())
    graphed_blocks = GraphedFunction(blocks, clear_rope_caches, active, weights)
    graphed_decoder = GraphedFunction(decoder, clear_rope_caches, active, weights)

    def encode_image(image, true_shape):
        x, pos = encoder.patch_embed(image, true_shape=true_shape)
        assert encoder.enc_pos_embed is None
        return graphed_blocks(x, pos)[0], pos, None

    encoder._encode_image = encode_image
    encoder._decoder = graphed_decoder
    return graphed_blocks, graphed_decoder
