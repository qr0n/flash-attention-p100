"""Parity check: does Qwen3-0.6B compute the same thing on the P100 kernels?

Runs one real text batch through the same weights three times -- torch `sdpa`,
torch `eager`, and `fa_p100` -- and compares logits and teacher-forced loss.

The yardstick that matters is not "is fa_p100 close to sdpa" in the abstract.
It is "is fa_p100-vs-sdpa within the envelope that eager-vs-sdpa already
occupies", because those two are both torch, both fp16, and disagree with each
other purely from summation order. A kernel inside that envelope is as correct
as the reference implementations are with respect to one another.

    env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python verify_hf.py
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fa_p100_hf import enable_fa_p100, register

MODEL = "Qwen/Qwen3-0.6B"


def load_text(path, tok, seq_len, batch):
    with open(path) as fh:
        raw = fh.read()
    ids = tok(raw, return_tensors="pt").input_ids[0]
    need = seq_len * batch
    if ids.numel() < need:
        raise SystemExit(f"corpus has {ids.numel()} tokens, need {need}")
    return ids[:need].view(batch, seq_len)


def run(model, impl, ids):
    model.config._attn_implementation = impl
    for mod in model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = impl
    with torch.no_grad():
        out = model(ids, use_cache=False)
    torch.cuda.synchronize()
    return out.logits.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--tol", type=float, default=2.0,
                    help="limit on fa_p100's MEAN logit deviation, as a "
                         "multiple of the eager-vs-sdpa deviation")
    ap.add_argument("--tail-tol", type=float, default=4.0,
                    help="same, for the p99.99 tail, which is legitimately "
                         "heavier than eager's")
    args = ap.parse_args()

    register()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa"
    ).cuda().eval()

    ids = load_text(args.corpus, tok, args.seq_len, args.batch).cuda()
    print(f"{MODEL}  batch={args.batch} seq_len={args.seq_len} "
          f"head_dim={model.config.head_dim} "
          f"heads={model.config.num_attention_heads}/"
          f"{model.config.num_key_value_heads} (GQA)\n")

    logits = {}
    for impl in ("sdpa", "eager", "fa_p100"):
        if impl == "fa_p100":
            enable_fa_p100(model)
        logits[impl] = run(model, impl, ids)

    def stats(a, b):
        """Robust deviation statistics.

        Deliberately NOT gated on max-abs. Over 2048 x 151936 logits the
        maximum is an extreme order statistic: eager's own max deviation from
        sdpa is a 2-in-311-million outlier, and the measured max ratio swings
        from 0.48x at seq_len 1024 to 3.89x at 2048 on an unchanged kernel.
        Mean and p99.99 are stable across both; those are what we gate on.
        """
        d = (a - b).abs()
        flat = d.flatten()[::13].sort().values      # subsample: 311M won't sort
        return {
            "mean": d.mean().item(),
            "p99": flat[int(flat.numel() * 0.99)].item(),
            "p9999": flat[int(flat.numel() * 0.9999)].item(),
            "max": d.max().item(),
            "over_0p2": (d > 0.2).sum().item(),
            "n": d.numel(),
        }

    def loss(lg):
        return torch.nn.functional.cross_entropy(
            lg[:, :-1].reshape(-1, lg.size(-1)), ids[:, 1:].reshape(-1)
        ).item()

    ref = logits["sdpa"]
    e = stats(logits["eager"], ref)
    f = stats(logits["fa_p100"], ref)

    print("logit deviation from sdpa")
    print(f"{'':<10}{'mean':>11}{'p99':>11}{'p99.99':>11}{'max':>11}   >0.2")
    for name, s in (("eager", e), ("fa_p100", f)):
        print(f"  {name:<8}{s['mean']:>11.3e}{s['p99']:>11.3e}"
              f"{s['p9999']:>11.3e}{s['max']:>11.3e}   {s['over_0p2']} of {s['n']}")
    print(f"  {'ratio':<8}{f['mean'] / e['mean']:>10.2f}x{f['p99'] / e['p99']:>10.2f}x"
          f"{f['p9999'] / e['p9999']:>10.2f}x{f['max'] / e['max']:>10.2f}x"
          "   (of the eager-vs-sdpa envelope)\n")

    print("teacher-forced cross-entropy on the same tokens")
    losses = {k: loss(v) for k, v in logits.items()}
    for k, v in losses.items():
        print(f"  {k:<8} {v:.6f}")
    print(f"  fa_p100 - sdpa = {losses['fa_p100'] - losses['sdpa']:+.6f} nats\n")

    # Rank agreement matters more than raw logit distance for a generative
    # model: a kernel that picks different argmaxes produces different text.
    agree = (logits["fa_p100"].argmax(-1) == ref.argmax(-1)).float().mean().item()
    agree_eager = (logits["eager"].argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"top-1 agreement with sdpa:  fa_p100 {agree:.4%}   eager {agree_eager:.4%}")

    # Four gates. Rank agreement and loss are what actually decide whether the
    # model behaves the same; mean and p99.99 bound the error distribution's
    # body and tail respectively.
    checks = [
        ("mean deviation", f["mean"] / e["mean"], args.tol,
         f"{f['mean']:.3e} vs eager {e['mean']:.3e}"),
        ("p99.99 deviation", f["p9999"] / e["p9999"], args.tail_tol,
         f"{f['p9999']:.3e} vs eager {e['p9999']:.3e}"),
    ]
    loss_gap = abs(losses["fa_p100"] - losses["sdpa"])
    agree_gap = agree_eager - agree

    print()
    ok = True
    for name, ratio, tol, detail in checks:
        good = ratio <= tol
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name:<18} {ratio:.2f}x "
              f"(limit {tol}x)   {detail}")
    good = loss_gap <= 1e-3
    ok &= good
    print(f"  [{'ok' if good else 'FAIL'}] {'loss gap':<18} {loss_gap:.2e} nats "
          f"(limit 1.0e-03)")
    good = agree_gap <= 0.01
    ok &= good
    print(f"  [{'ok' if good else 'FAIL'}] {'top-1 vs eager':<18} "
          f"{agree_gap:+.4%} (limit 1%)")

    print(f"\nmax-abs is reported but NOT gated: it is a 1-in-{f['n']} order "
          f"statistic and swings with seq_len on an unchanged kernel.")
    print(("PASS" if ok else "FAIL") + ": fa_p100 tracks sdpa within the "
          "envelope torch's own backends occupy")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
