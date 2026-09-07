"""Fine-tune Qwen3-0.6B on the P100 kernels, and on torch SDPA, and compare.

This is the thing HANDOFF.md section 5A said was missing: a real caller. It runs
the *same* fine-tune twice from the same initial weights over the same data in
the same order, once per attention implementation, and reports both the loss
trajectory (does the kernel train the model correctly?) and the wall clock
(does it train it faster?).

Sequences are packed into fixed-length blocks, so every position is a real
token and no padding mask is ever needed -- which is what lets the kernel be
used at all, since it takes no mask.

Precision: fp32 master weights, `torch.autocast` fp16 for the forward, and a
`GradScaler`. Pascal has no bf16 hardware, so fp16 is not a preference here.
Note that q and k still reach the kernel as fp32 even under autocast, because
Qwen3's q_norm/k_norm multiply by an fp32 master weight; fa_p100_hf casts them
down explicitly, which is what torch's SDPA gets for free from autocast.

    env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python finetune_hf.py \
        --corpus /path/to/corpus.txt --steps 30
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fa_p100_hf import enable_fa_p100, register

MODEL = "Qwen/Qwen3-0.6B"


def build_blocks(path, tok, seq_len, n_blocks):
    """Pack the corpus into (n_blocks, seq_len) with no padding anywhere."""
    with open(path) as fh:
        ids = tok(fh.read(), return_tensors="pt").input_ids[0]
    need = seq_len * n_blocks
    if ids.numel() < need:
        raise SystemExit(f"corpus has {ids.numel()} tokens, need {need}")
    return ids[:need].view(n_blocks, seq_len)


def chunked_loss(model, ids, chunks):
    """Cross-entropy without ever materialising the full logit tensor.

    Qwen3-0.6B has a 151936-entry vocabulary, so at seq_len 2048 the fp32
    logits alone are 1.24 GiB and `labels=` OOMs a 16 GiB card before
    attention is even a factor. Checkpointing the head per chunk recomputes
    the logits in the backward instead of storing them, which is what makes
    seq_len 2048+ reachable at all -- and therefore what makes the attention
    kernel's contribution measurable rather than lost behind the head.
    """
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
    shifted, labels = hidden[:, :-1], ids[:, 1:]
    n = labels.numel()

    def head_loss(h, lab):
        logits = model.lm_head(h).float()
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), lab.reshape(-1), reduction="sum"
        )

    total = 0.0
    for h_c, l_c in zip(shifted.chunk(chunks, dim=1), labels.chunk(chunks, dim=1)):
        total = total + torch.utils.checkpoint.checkpoint(
            head_loss, h_c, l_c, use_reentrant=False
        )
    return total / n


def make_model(impl, full_precision, grad_checkpointing):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="sdpa"
    ).cuda()
    model.config.use_cache = False
    if grad_checkpointing:
        # How long-context fine-tuning actually fits on a 16 GiB card. It also
        # re-runs the forward during the backward, so attention is executed
        # twice per step and its share of the wall clock rises.
        model.gradient_checkpointing_enable()
    if impl == "fa_p100":
        enable_fa_p100(model, full_precision=full_precision)
    model.train()
    return model


def train(impl, blocks, args):
    model = make_model(impl, args.full_precision, args.grad_checkpointing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()

    losses, step_times = [], []
    for step in range(args.steps):
        batch = blocks[step % blocks.size(0)].unsqueeze(0).cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.autocast("cuda", dtype=torch.float16):
            loss = chunked_loss(model, batch, args.loss_chunks)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        losses.append(loss.item())
        # The first steps include autograd graph setup and, for fa_p100, the
        # kernel's first-call JIT; they are not representative.
        if step >= args.warmup:
            step_times.append(dt)
        print(f"  [{impl}] step {step:3d}  loss {losses[-1]:.4f}  {dt * 1e3:8.1f} ms")

    peak = torch.cuda.max_memory_allocated() / 2**30
    del model, opt, scaler
    torch.cuda.empty_cache()
    return {
        "impl": impl,
        "losses": losses,
        "median_step_ms": sorted(step_times)[len(step_times) // 2] * 1e3,
        "peak_gib": peak,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--loss-chunks", type=int, default=8)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--full-precision", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    register()
    tok = AutoTokenizer.from_pretrained(MODEL)
    blocks = build_blocks(args.corpus, tok, args.seq_len, args.blocks)
    print(f"{MODEL}: {args.blocks} packed blocks of {args.seq_len} tokens, "
          f"{args.steps} steps, no padding\n")

    results = [train(impl, blocks, args) for impl in ("sdpa", "fa_p100")]
    sdpa, flash = results

    print("\n" + "=" * 62)
    print(f"{'':<12}{'median step':>14}{'peak mem':>12}{'final loss':>14}")
    for r in results:
        print(f"{r['impl']:<12}{r['median_step_ms']:>11.1f} ms"
              f"{r['peak_gib']:>9.2f} GiB{r['losses'][-1]:>14.4f}")
    print(f"\nspeedup       {sdpa['median_step_ms'] / flash['median_step_ms']:.2f}x"
          f"   memory {flash['peak_gib'] / sdpa['peak_gib']:.2f}x of sdpa")

    drift = max(abs(a - b) for a, b in zip(sdpa["losses"], flash["losses"]))
    print(f"max |loss difference| across the run: {drift:.2e}")
    print("(same init, same data order -- the two curves should track; any "
          "gap is fp16 summation order, not a different computation)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"args": vars(args), "results": results}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
