"""Phase 5 acceptance: does the binding work, is it correct, is it faster."""
import math
import time

import torch
import torch.nn.functional as F

import fa_p100


def timed(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def rel(a, b):
    """Max error relative to the RMS of the reference -- the same yardstick
    BENCH.log uses, so numbers here are comparable to the CUDA-side ones."""
    b = b.float()
    return ((a.float() - b).abs().max() / b.pow(2).mean().sqrt()).item()


def main():
    print(f"torch {torch.__version__}")
    print(f"arch list: {torch.cuda.get_arch_list()}")
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"capability {torch.cuda.get_device_capability(0)}")
    if "sm_60" not in torch.cuda.get_arch_list():
        print("\n*** sm_60 is NOT in the arch list -- no kernel will run. ***")
        return

    torch.manual_seed(0)
    dev = "cuda"

    print("\n=== forward correctness (vs plain-torch fp32 reference) ===")
    print(f"{'shape':>22} {'causal':>7} {'fp16 tier':>10} {'fp32 tier':>10}")
    for (B, H, N) in [(1, 4, 256), (2, 8, 512), (1, 16, 1024)]:
        q, k, v = (torch.randn(B, H, N, 64, device=dev, dtype=torch.half) for _ in range(3))
        for causal in (False, True):
            ref = fa_p100.reference(q, k, v, causal)
            e16 = rel(fa_p100.flash_attn(q, k, v, causal, False), ref)
            e32 = rel(fa_p100.flash_attn(q, k, v, causal, True), ref)
            print(f"{f'({B},{H},{N},64)':>22} {str(causal):>7} {e16:10.2e} {e32:10.2e}")

    print("\n=== gradient correctness (vs autograd through the reference) ===")
    print(f"{'shape':>22} {'causal':>7} {'dQ':>10} {'dK':>10} {'dV':>10}")
    for (B, H, N) in [(1, 4, 256), (2, 4, 512)]:
        for causal in (False, True):
            base = [torch.randn(B, H, N, 64, device=dev, dtype=torch.half) for _ in range(3)]
            qa, ka, va = (t.clone().requires_grad_(True) for t in base)
            qb, kb, vb = (t.clone().requires_grad_(True) for t in base)
            g = torch.randn(B, H, N, 64, device=dev, dtype=torch.half)

            fa_p100.flash_attn(qa, ka, va, causal, True).backward(g)
            fa_p100.reference(qb, kb, vb, causal).backward(g)
            print(f"{f'({B},{H},{N},64)':>22} {str(causal):>7} "
                  f"{rel(qa.grad, qb.grad):10.2e} {rel(ka.grad, kb.grad):10.2e} "
                  f"{rel(va.grad, vb.grad):10.2e}")

    print("\n=== speed vs torch scaled_dot_product_attention ===")
    print(f"{'shape':>22} {'causal':>7} {'sdpa ms':>9} {'flash ms':>9} {'speedup':>8} {'peak MB':>9}")
    for (B, H, N) in [(1, 16, 2048), (1, 16, 4096)]:
        q, k, v = (torch.randn(B, H, N, 64, device=dev, dtype=torch.half) for _ in range(3))
        for causal in (False, True):
            torch.cuda.reset_peak_memory_stats()
            t_sdpa = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
            m_sdpa = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()
            t_fa = timed(lambda: fa_p100.flash_attn(q, k, v, causal, False))
            m_fa = torch.cuda.max_memory_allocated() / 1e6
            print(f"{f'({B},{H},{N},64)':>22} {str(causal):>7} {t_sdpa:9.3f} {t_fa:9.3f} "
                  f"{t_sdpa / t_fa:7.2f}x  {m_sdpa:.0f} -> {m_fa:.0f}")

    print("\n=== training-step smoke test (fwd + bwd, causal) ===")
    B, H, N = 1, 8, 1024
    q, k, v = (torch.randn(B, H, N, 64, device=dev, dtype=torch.half, requires_grad=True)
               for _ in range(3))
    g = torch.randn(B, H, N, 64, device=dev, dtype=torch.half)

    def step():
        for t in (q, k, v):
            t.grad = None
        fa_p100.flash_attn(q, k, v, True, False).backward(g)

    print(f"  fwd+bwd {timed(step):.3f} ms   grads finite: "
          f"{all(t.grad.isfinite().all().item() for t in (q, k, v))}")


if __name__ == "__main__":
    main()
