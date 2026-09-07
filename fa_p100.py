"""FlashAttention for 2x Tesla P100 (sm_60) as a torch autograd Function.

The kernels are hand-written SIMT half2 -- there are no tensor cores on Pascal.
Built on demand with torch.utils.cpp_extension; the first import compiles.

    from fa_p100 import flash_attn
    o = flash_attn(q, k, v, causal=True)      # q,k,v: (B, H, N, 64) fp16 CUDA

`full_precision=True` switches the forward's QK^T to fp32 MACs. It costs ~28%
and is what makes the kernel exact for unnormalised inputs; the default fp16
tier is fine for LayerNorm'd O(1) activations. The backward is always fp32.
"""
import math
import os

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))

_ext = load(
    name="fa_p100_ext",
    sources=[os.path.join(_HERE, "fa_p100_ext.cu")],
    extra_cuda_cflags=["-arch=sm_60", "-O3", "--use_fast_math"],
    extra_include_paths=[_HERE],
    extra_ldflags=["-lcublas"],
    verbose=False,
)


class _FlashAttnP100(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, full_precision):
        o, lse = _ext.forward(q, k, v, causal, full_precision)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, grad_o):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = _ext.backward(q, k, v, grad_o.contiguous(), o, lse, ctx.causal)
        # The kernels accumulate gradients in fp32; hand them back in the
        # parameters' own dtype.
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def flash_attn(q, k, v, causal=False, full_precision=False):
    """Attention over fp16 CUDA tensors.

    q is (B, Hq, N, 64); k and v are (B, Hkv, N, 64) with Hq a multiple of Hkv.
    Hkv == Hq is ordinary multi-head; Hkv == 1 is multi-query; anything between
    is grouped-query. N is arbitrary.
    """
    return _FlashAttnP100.apply(q, k, v, causal, full_precision)


def _expand_kv(k, n_head):
    """Repeat each K/V head across the query heads that share it."""
    if k.size(1) == n_head:
        return k
    return k.repeat_interleave(n_head // k.size(1), dim=1)


def reference(q, k, v, causal=False):
    """Same computation in plain torch, for comparison. Handles GQA/MQA."""
    k = _expand_kv(k, q.size(1))
    v = _expand_kv(v, q.size(1))
    scale = 1.0 / math.sqrt(q.size(-1))
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    if causal:
        n = q.size(-2)
        mask = torch.triu(torch.ones(n, n, device=q.device, dtype=torch.bool), 1)
        s = s.masked_fill(mask, float("-inf"))
    return (s.softmax(-1) @ v.float()).to(q.dtype)
