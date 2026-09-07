"""Honest end-to-end comparison: forward AND backward, against torch SDPA.

Phase 5 only ever timed the forward. Training pays for both, and this kernel's
backward is its weak half, so the fwd-only speedup is not the number that
matters for anyone actually training.
"""
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


def bench(B, H, N, causal):
    mk = lambda: torch.randn(B, H, N, 64, device="cuda", dtype=torch.half,
                             requires_grad=True)
    q, k, v = mk(), mk(), mk()
    g = torch.randn(B, H, N, 64, device="cuda", dtype=torch.half)

    def clear():
        for t in (q, k, v):
            t.grad = None

    # forward only (graph built, nothing backwarded)
    f_torch = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
    f_mine  = timed(lambda: fa_p100.flash_attn(q, k, v, causal, False))

    # forward + backward
    def step_torch():
        clear()
        F.scaled_dot_product_attention(q, k, v, is_causal=causal).backward(g)

    def step_mine():
        clear()
        fa_p100.flash_attn(q, k, v, causal, False).backward(g)

    t_torch = timed(step_torch)
    t_mine  = timed(step_mine)

    # peak memory over a full training step
    torch.cuda.reset_peak_memory_stats(); step_torch(); torch.cuda.synchronize()
    m_torch = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats(); step_mine(); torch.cuda.synchronize()
    m_mine = torch.cuda.max_memory_allocated() / 1e6

    return dict(f_torch=f_torch, f_mine=f_mine,
                b_torch=t_torch - f_torch, b_mine=t_mine - f_mine,
                t_torch=t_torch, t_mine=t_mine, m_torch=m_torch, m_mine=m_mine)


def main():
    print(f"torch {torch.__version__}  |  {torch.cuda.get_device_name(0)}")
    print("baseline = F.scaled_dot_product_attention (mem_efficient backend on sm_60)\n")
    hdr = (f"{'shape':>16} {'caus':>5} | {'fwd torch':>9} {'fwd mine':>9} {'x':>6} "
           f"| {'bwd torch':>9} {'bwd mine':>9} {'x':>6} "
           f"| {'STEP torch':>10} {'STEP mine':>10} {'x':>6} | {'peak MB':>13}")
    print(hdr)
    print("-" * len(hdr))
    for (B, H, N) in [(1, 16, 2048), (1, 16, 4096)]:
        for causal in (False, True):
            r = bench(B, H, N, causal)
            print(f"{f'{B},{H},{N}':>16} {str(causal):>5} | "
                  f"{r['f_torch']:9.3f} {r['f_mine']:9.3f} {r['f_torch']/r['f_mine']:5.2f}x | "
                  f"{r['b_torch']:9.3f} {r['b_mine']:9.3f} {r['b_torch']/r['b_mine']:5.2f}x | "
                  f"{r['t_torch']:10.3f} {r['t_mine']:10.3f} "
                  f"{r['t_torch']/r['t_mine']:5.2f}x | "
                  f"{r['m_torch']:5.0f} -> {r['m_mine']:5.0f}")


if __name__ == "__main__":
    main()
