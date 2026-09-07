"""Drop-in FlashAttention-P100 for HuggingFace transformers.

Registers the sm_60 kernels as an attention implementation named `fa_p100`, so
any supported model can be run with

    from fa_p100_hf import enable_fa_p100
    model = AutoModelForCausalLM.from_pretrained(..., dtype=torch.float16)
    enable_fa_p100(model)

transformers is pinned to 4.x here: 5.x requires torch>=2.5, and torch is
pinned to 2.4.1 because it is the last build shipping sm_60. See HANDOFF.md.

WHY THE MASK REGISTRY IS LEFT ALONE
-----------------------------------
`masking_utils._preprocess_mask_arguments` early-exits with a None mask for any
`_attn_implementation` that is absent from `ALL_MASK_ATTENTION_FUNCTIONS`. We
want exactly that -- no B x H x N x N mask is ever materialised, and the kernel
applies causality itself. The cost is that a padding mask handed to
`model(...)` would be dropped *silently*, so `enable_fa_p100` installs a
forward guard that raises instead. Do not "fix" this by registering a mask
function: the kernel takes no mask argument, so the mask would be built and
then ignored, which is the same bug with more memory traffic.

WHAT IS NOT SUPPORTED -- all of these raise, none of them degrade quietly
------------------------------------------------------------------------
  * padding / arbitrary masks   -- pack sequences to a fixed length instead
  * sliding-window attention    -- kernel is dense or causal only
  * attention dropout           -- kernel has no RNG
  * a scale other than 1/sqrt(D) -- it is hardcoded in the .cu dispatch
  * head_dim other than 64/128
  * incremental decoding with a KV cache (q_len != kv_len)
  * output_attentions           -- the probabilities are never materialised
"""

import math
import warnings

import torch

from fa_p100 import flash_attn

_IMPL_NAME = "fa_p100"
_warned_cast = False


def fa_p100_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    dropout: float = 0.0,
    scaling=None,
    is_causal=None,
    **kwargs,
):
    """transformers attention-interface entry point.

    Signature and return contract mirror `sdpa_attention_forward`: tensors
    arrive as (B, H, N, D) and the output goes back as (B, N, H, D).
    """
    global _warned_cast

    if attention_mask is not None:
        raise NotImplementedError(
            "fa_p100 received an attention mask. The kernel takes no mask "
            "argument. Pack your sequences to a fixed length so no padding is "
            "needed; see enable_fa_p100's forward guard."
        )
    if kwargs.get("output_attentions", False):
        raise NotImplementedError(
            "fa_p100 never materialises the attention probabilities, so "
            "output_attentions=True cannot be served. Use attn_implementation="
            "'eager' for that."
        )
    if kwargs.get("head_mask") is not None:
        raise NotImplementedError("fa_p100 does not support head_mask.")
    if kwargs.get("sliding_window") is not None:
        raise NotImplementedError(
            "fa_p100 is dense-or-causal only; layer requests a sliding window "
            f"of {kwargs['sliding_window']}."
        )
    if dropout:
        raise NotImplementedError(
            f"fa_p100 has no dropout path (got attention_dropout={dropout})."
        )

    head_dim = query.size(-1)
    if head_dim not in (64, 128):
        raise NotImplementedError(
            f"fa_p100 supports head_dim 64 or 128, got {head_dim}."
        )
    # The .cu dispatch hardcodes scale = 1/sqrt(D); a model wanting anything
    # else would be silently mis-scaled.
    if scaling is not None and abs(scaling - head_dim ** -0.5) > 1e-6:
        raise NotImplementedError(
            f"fa_p100 hardcodes scale=1/sqrt(D)={head_dim ** -0.5:.6g}, "
            f"model asked for {scaling:.6g}."
        )
    if key.size(2) != query.size(2):
        raise NotImplementedError(
            f"fa_p100 requires q and kv to have the same length, got "
            f"q={query.size(2)} kv={key.size(2)}. This is the incremental-"
            "decoding path (use_cache=True); the kernel is training-only."
        )

    if is_causal is None:
        is_causal = getattr(module, "is_causal", True) and query.size(2) > 1

    # Pascal has no bf16 hardware and the kernel is fp16-only.
    #
    # Casting here is the NORMAL path under autocast, not a fallback. Qwen3's
    # per-head q_norm/k_norm multiply by an fp32 master weight, which promotes
    # q and k back to fp32 after the fp16 q_proj; the rotary cos/sin are fp32
    # for the same reason. So attention is reached with q,k fp32 and v fp16.
    # torch's own SDPA hides this because autocast intercepts it and casts the
    # inputs down; a custom interface has to do it explicitly. Cast quietly
    # when autocast asked for fp16, and warn only when the cast really is
    # unexpected.
    orig_dtype = query.dtype
    if query.dtype != torch.float16 or key.dtype != torch.float16 \
            or value.dtype != torch.float16:
        sanctioned = (torch.is_autocast_enabled("cuda")
                      and torch.get_autocast_dtype("cuda") == torch.float16)
        if not sanctioned and not _warned_cast:
            warnings.warn(
                f"fa_p100 casting q={query.dtype} k={key.dtype} v={value.dtype} "
                "-> float16 outside autocast; the kernel is fp16 only. Wrap "
                "the step in torch.autocast('cuda', torch.float16).",
                stacklevel=2,
            )
            _warned_cast = True
        query, key, value = query.half(), key.half(), value.half()

    # No repeat_kv: the kernel consumes GQA/MQA natively, so the expanded
    # K/V never exist. sdpa_attention_forward has to materialise them
    # whenever a mask is present.
    full_precision = getattr(module, "_fa_p100_full_precision", False)
    out = flash_attn(query, key, value, causal=bool(is_causal),
                     full_precision=full_precision)

    return out.transpose(1, 2).contiguous().to(orig_dtype), None


def register():
    """Add `fa_p100` to the transformers attention registry (idempotent)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS[_IMPL_NAME] = fa_p100_attention_forward
    return _IMPL_NAME


def _set_impl(config, name):
    config._attn_implementation = name
    for sub in getattr(config, "sub_configs", {}) or {}:
        child = getattr(config, sub, None)
        if child is not None:
            _set_impl(child, name)


def enable_fa_p100(model, full_precision=False):
    """Switch `model` onto the P100 kernels and guard the unsupported paths.

    `full_precision` selects the forward's fp32-MAC tier. The default fp16
    tier is correct for LayerNorm'd O(1) activations, which is what Qwen3's
    per-head q_norm/k_norm produce; set it True for models that feed the
    kernel unnormalised Q/K.
    """
    register()
    _set_impl(model.config, _IMPL_NAME)
    for mod in model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            _set_impl(mod.config, _IMPL_NAME)
        mod._fa_p100_full_precision = full_precision

    # The kernel is training-only: it requires q_len == kv_len, so a populated
    # KV cache breaks it. Default the cache off rather than let a plain
    # forward build one it cannot use.
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False

    if not getattr(model, "_fa_p100_guarded", False):
        inner = model.forward

        def guarded(*args, attention_mask=None, use_cache=None, **kwargs):
            # transformers drops the mask before it reaches the attention
            # interface for unregistered impls, so this is the only place a
            # padding mask can still be caught.
            if attention_mask is not None and not bool(attention_mask.all()):
                raise NotImplementedError(
                    "fa_p100 cannot honour a padding mask, and transformers "
                    "would discard it silently. Pack the batch to a fixed "
                    "length so every position is real."
                )
            # Caught here rather than in the attention interface, where it
            # would only fire on the first decode step -- i.e. after a
            # successful prefill, halfway through a generate() call.
            if use_cache:
                raise NotImplementedError(
                    "fa_p100 requires q_len == kv_len, so it cannot serve a "
                    "populated KV cache; it is a training kernel. Prefill "
                    "would succeed and the first decode step would fail. For "
                    "cache-free (quadratic, but correct) generation set "
                    "generation_config.use_cache = False."
                )
            return inner(*args, attention_mask=attention_mask,
                         use_cache=use_cache, **kwargs)

        model.forward = guarded
        model._fa_p100_guarded = True

    return model
