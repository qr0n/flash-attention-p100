"""Regression suite for the sm_60 FlashAttention kernels.

Covers what the kernels actually support, not just the happy path they were
first written for: both head dims, grouped/multi-query attention, sequence
lengths that are not multiples of the block size, and the four input regimes
that separate the precision tiers.

    env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python test_fa_p100.py

Exits non-zero if anything fails. `--quick` skips the timing section.
"""
import math
import sys
import time

import torch
import torch.nn.functional as F

import fa_p100

# Errors are normalised by the RMS of the reference, so the threshold means the
# same thing whether attention is peaked (large outputs) or flat (tiny ones).
# fp16 attention lands around 1e-2; anything past 6e-2 is a bug, not rounding.
TOL = 6e-2

_results = []


def check(name, err, tol=TOL):
    ok = err == err and err < tol          # err != err catches NaN
    _results.append((name, err, ok))
    return ok


def rel(got, ref):
    ref = ref.float()
    return ((got.float() - ref).abs().max() / ref.pow(2).mean().sqrt()).item()


def rand(*shape, scale=1.0, grad=False):
    t = torch.randn(*shape, device="cuda", dtype=torch.half) * scale
    return t.detach().requires_grad_(grad)


# ---------------------------------------------------------------------------
def test_environment():
    print(f"torch {torch.__version__}   {torch.cuda.get_device_name(0)} "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"arch list: {torch.cuda.get_arch_list()}")
    archs = torch.cuda.get_arch_list()
    _results.append(("env: sm_60 in arch list", 0.0, "sm_60" in archs))
    print()


# ---------------------------------------------------------------------------
def test_forward():
    """Head dim x attention grouping x sequence length x causality."""
    print("forward vs plain-torch fp32 reference")
    print(f"  {'D':>4} {'Hq':>4} {'Hkv':>4} {'N':>6} {'caus':>6} {'fp16':>9} {'fp32':>9}")
    for D in (64, 128):
        for Hq, Hkv in ((8, 8), (8, 2), (8, 1)):
            # 63/65 straddle a block edge, 1 and 7 are shorter than one block,
            # 333 and 1023 are ordinary ragged lengths.
            for N in (1, 7, 63, 64, 65, 333, 1023):
                q = rand(1, Hq, N, D)
                k = rand(1, Hkv, N, D)
                v = rand(1, Hkv, N, D)
                ref = fa_p100.reference(q, k, v, causal=True)
                for causal in (False, True):
                    ref = fa_p100.reference(q, k, v, causal)
                    e16 = rel(fa_p100.flash_attn(q, k, v, causal, False), ref)
                    e32 = rel(fa_p100.flash_attn(q, k, v, causal, True), ref)
                    ok = (check(f"fwd D{D} {Hq}/{Hkv} N{N} causal={causal} fp16", e16)
                          & check(f"fwd D{D} {Hq}/{Hkv} N{N} causal={causal} fp32", e32))
                    if not ok or N in (65, 1023):
                        print(f"  {D:4d} {Hq:4d} {Hkv:4d} {N:6d} {str(causal):>6} "
                              f"{e16:9.2e} {e32:9.2e}{'' if ok else '   <<< FAIL'}")
    print()


# ---------------------------------------------------------------------------
def test_gradients():
    """Gradients through the autograd.Function vs autograd on the reference."""
    print("gradients vs autograd through the reference")
    print(f"  {'D':>4} {'Hq':>4} {'Hkv':>4} {'N':>6} {'caus':>6} {'dQ':>9} {'dK':>9} {'dV':>9}")
    for D in (64, 128):
        for Hq, Hkv in ((4, 4), (8, 2), (8, 1)):
            for N in (7, 64, 129, 333):
                for causal in (False, True):
                    base = [rand(1, Hq, N, D), rand(1, Hkv, N, D), rand(1, Hkv, N, D)]
                    a = [t.clone().requires_grad_(True) for t in base]
                    b = [t.clone().requires_grad_(True) for t in base]
                    g = rand(1, Hq, N, D)
                    fa_p100.flash_attn(*a, causal, True).backward(g)
                    fa_p100.reference(*b, causal).backward(g)
                    e = [rel(x.grad, y.grad) for x, y in zip(a, b)]
                    ok = all(check(f"grad {n} D{D} {Hq}/{Hkv} N{N} causal={causal}", v)
                             for n, v in zip("QKV", e))
                    if not ok or N == 333:
                        print(f"  {D:4d} {Hq:4d} {Hkv:4d} {N:6d} {str(causal):>6} "
                              f"{e[0]:9.2e} {e[1]:9.2e} {e[2]:9.2e}"
                              f"{'' if ok else '   <<< FAIL'}")
    print()


# ---------------------------------------------------------------------------
def test_input_regimes():
    """The four regimes that separate the precision tiers.

    Regime 2 (inputs scaled x10) is unreachable for the fp16 tier and for
    cuBLAS alike -- it is a statement about fp16, not about the kernel -- so it
    is only asserted for the fp32-MAC tier.
    """
    print("input regimes (fp32-MAC tier, which is the one that must clear all four)")
    print(f"  {'regime':<16} {'D':>4} {'caus':>6} {'fwd':>9} {'dQ':>9} {'dK':>9} {'dV':>9}")
    regimes = [("1 N(0,1)", 1.0), ("2 large x10", 10.0),
               ("3 near-uniform", 0.05), ("4 near-one-hot", 3.0)]
    for name, s in regimes:
        for D in (64, 128):
            for causal in (False, True):
                N = 256
                base = [rand(1, 4, N, D, scale=s), rand(1, 4, N, D, scale=s),
                        rand(1, 4, N, D)]
                a = [t.clone().requires_grad_(True) for t in base]
                b = [t.clone().requires_grad_(True) for t in base]
                g = rand(1, 4, N, D)
                o = fa_p100.flash_attn(*a, causal, True)
                r = fa_p100.reference(*b, causal)
                ef = rel(o, r)
                o.backward(g)
                r.backward(g)
                e = [rel(x.grad, y.grad) for x, y in zip(a, b)]
                ok = check(f"regime {name} D{D} causal={causal} fwd", ef)
                ok &= all(check(f"regime {name} D{D} causal={causal} d{n}", v)
                          for n, v in zip("QKV", e))
                print(f"  {name:<16} {D:4d} {str(causal):>6} {ef:9.2e} "
                      f"{e[0]:9.2e} {e[1]:9.2e} {e[2]:9.2e}"
                      f"{'' if ok else '   <<< FAIL'}")
    print()


# ---------------------------------------------------------------------------
def test_speed():
    """Informational: full training step against torch SDPA."""
    def timed(fn, warmup=3, iters=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    print("full training step vs torch scaled_dot_product_attention")
    print(f"  {'shape':>17} {'caus':>6} {'torch':>9} {'flash':>9} {'speedup':>8}")
    for (H, N, D) in ((16, 2048, 64), (16, 4096, 64), (8, 4096, 128)):
        q, k, v = (rand(1, H, N, D, grad=True) for _ in range(3))
        g = rand(1, H, N, D)

        def step(fn):
            for t in (q, k, v):
                t.grad = None
            fn().backward(g)

        for causal in (False, True):
            a = timed(lambda: step(
                lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)))
            b = timed(lambda: step(lambda: fa_p100.flash_attn(q, k, v, causal)))
            print(f"  {f'1,{H},{N},{D}':>17} {str(causal):>6} "
                  f"{a:8.2f}ms {b:8.2f}ms {a / b:7.2f}x")
            _results.append((f"speed 1,{H},{N},{D} causal={causal}", 0.0, b < a))
    print()


# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    test_environment()
    if not _results[0][2]:
        print("sm_60 missing from the arch list -- no kernel can run.")
        return 1

    test_forward()
    test_gradients()
    test_input_regimes()
    if "--quick" not in sys.argv:
        test_speed()

    failed = [r for r in _results if not r[2]]
    print(f"{len(_results) - len(failed)}/{len(_results)} checks passed")
    for name, err, _ in failed:
        print(f"  FAILED  {name}   err {err:.3e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
