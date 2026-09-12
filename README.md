# FlashAttention for Tesla P100

A from-scratch FlashAttention-2 forward **and backward** for the NVIDIA Tesla
P100 (GP100, `sm_60`) — a GPU with **no tensor cores**, no `cp.async`, no
`ldmatrix`, and a hard 48 KB shared-memory-per-block cap.

Every matmul is a register-blocked `__hfma2` outer product written by hand.

**It is 3.1–3.9× faster than `torch.nn.functional.scaled_dot_product_attention`
on a full training step**, and it runs a stock HuggingFace model.

---

## Why this exists

The official `flash-attn` package requires Ampere or newer. On Pascal, PyTorch's
SDPA silently falls back to its `mem_efficient` backend — `flash` reports
*"No available kernel"*. P100s are cheap and plentiful second-hand, and they have
one genuinely good property: **full-rate `half2` arithmetic** (19.04 TFLOPS fp16,
2× its fp32). Nothing was using it.

So the baseline here is not a strawman. It is torch's real, competent
`mem_efficient` kernel, which already avoids materialising the N² score matrix.

## Results

Full training step (forward + backward), fp16, vs `torch.nn.F.scaled_dot_product_attention`:

| shape `(B,H,N,D)` | causal | torch | this kernel | speedup |
|---|---|---|---|---|
| `1,16,2048,64` | yes | 25.12 ms | **8.07 ms** | **3.11×** |
| `1,16,4096,64` | yes | 97.35 ms | **29.60 ms** | **3.29×** |
| `1,8,2048,128` | yes | 45.34 ms | **12.04 ms** | **3.76×** |
| `1,8,4096,128` | yes | 176.32 ms | **45.49 ms** | **3.88×** |

**The backward is the bigger win, not the forward** — at `N=4096` causal, the
forward is 2.15× and the backward is 3.46×. Peak memory is ~10% lower.

### Driving a real model

`fa_p100_hf.py` registers the kernels as a HuggingFace attention implementation,
so an **unmodified** `Qwen/Qwen3-0.6B` runs on them:

| seq_len | torch sdpa | fa_p100 | speedup |
|---|---|---|---|
| 1024 | 1217.7 ms | 1061.6 ms | 1.15× |
| 2048 | 2934.9 ms | 2288.9 ms | 1.28× |
| 4096 | 9486.6 ms | 6665.6 ms | 1.42× |
| 8192 | 29366.0 ms | 18117.8 ms | **1.62×** |

The kernel-level 3.9× becomes 1.15–1.62× end to end, and that gap is Amdahl, not
a caveat: a 0.6B model with a 151936-entry vocabulary spends most of its step in
the LM head and the MLPs. Attention's share grows with sequence length — which is
exactly the shape of that column.

**Correctness** is measured against the envelope torch's *own* `eager` and `sdpa`
backends occupy when they disagree with each other from summation order alone:
mean logit deviation **1.01–1.03×** of that envelope, loss gap ~2.5e-4 nats,
top-1 agreement 99.41% against eager's 99.61%. Deviation is flat across sequence
position, so the online softmax accumulates no drift as N grows.

## What it supports

| | |
|---|---|
| head dim | **64 or 128** |
| sequence length | **arbitrary** — ragged tails handled, no padding required |
| attention | MHA, **GQA, MQA** (`Hq` any multiple of `Hkv`) |
| causal | yes, and it skips the upper triangle rather than masking it |
| precision | fp16 in/out, fp32 gradient accumulation; optional fp32 MACs in the forward |

**Not supported** — and every one of these *raises* rather than degrading
quietly: dropout, attention bias / ALiBi, arbitrary or padding masks,
sliding-window and block-sparse patterns, head dims other than 64/128, KV-cache
decoding (this is a training kernel), and bf16 — Pascal has no hardware for it.

## Requirements

- Tesla P100 / GP100 (`sm_60`). Other Pascals (P40, P4 — `sm_61`) run fp16 at
  **quarter** rate and would be slower than fp32 here.
- **CUDA 12.x** — CUDA 13 dropped Pascal entirely.
- **PyTorch 2.4.1+cu121** — the last build shipping `sm_60` kernels.
- `transformers` 4.x for the HuggingFace path (5.x requires torch ≥ 2.5).

## Quick start

```bash
make                                    # standalone CUDA binaries
env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python test_fa_p100.py
```

> The `PATH` prefix is **not optional**. `torch.utils.cpp_extension` shells out to
> the **ninja binary**, and `.venv/bin` only lands on `PATH` when the venv is
> activated. `import ninja` succeeding proves nothing.

Direct use:

```python
from fa_p100 import flash_attn

# q: (B, Hq, N, D)   k, v: (B, Hkv, N, D)   D in {64, 128}, N arbitrary
o = flash_attn(q, k, v, causal=True)
```

On a HuggingFace model:

```python
import torch
from transformers import AutoModelForCausalLM
from fa_p100_hf import enable_fa_p100

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").cuda()
enable_fa_p100(model)     # every unsupported path raises; none degrade quietly
```

## Layout

| file | what |
|---|---|
| `flash_fwd.cuh` | forward kernel, templated on `BR, BC, D, THREADS, CAUSAL, DRAIN, FP32MAC, BOUNDS` |
| `flash_bwd.cuh` | backward, adds `NO_ATOMIC`, `QDRAIN`, `GOutT`, `H2_DVDK` |
| `fa_p100_ext.cu`, `fa_p100.py` | torch binding + `autograd.Function` |
| `fa_p100_hf.py` | HuggingFace attention implementation, with guards |
| `test_fa_p100.py` | **383-check regression suite**, non-zero exit on failure |
| `bench_train.py` | full training step vs torch SDPA |
| `verify_hf.py`, `finetune_hf.py` | parity and fine-tune comparison on a real model |
| `gemm.cu`, `bench_cublas.cu`, `attention.cu`, `bwd.cu`, `multigpu.cu` | standalone CUDA drivers and baselines |

## Documentation

- **[`REPORT.md`](REPORT.md)** — what was built and measured, including where the
  original plan turned out to be wrong.
- **[`HANDOFF.md`](HANDOFF.md)** — what to do next, and the optimisations that
  were tried and **failed**, with numbers. Read §3 before optimising anything.
- **`BENCH.log`** — append-only measurement record.

One result worth stating up front, from `HANDOFF.md` §3: `__hfma2` is a **+13%
win in the dV/dK contraction and ruinous in the backward's outer products**
(accuracy blows out to ~1e0). Same instruction, opposite verdict, depending only
on which matmul it lands in — because `dS = P(dP − D)` is a cancellation and
`dV = PᵀdO` is not. Decide per matmul, not per kernel.

## Measured hardware ceiling

| | spec | measured |
|---|---|---|
| fp16 packed `half2` | 19.04 TFLOPS | **15.79** |
| fp32 FMA | 9.52 TFLOPS | **8.68** |
| HBM2, read+write (hardest pattern) | 732 GB/s | **498.8** |
| HBM2, pure read (the real ceiling) | 732 GB/s | **607** |

The kernels reach ~83% of spec fp16, so there is not much left on the table at
the instruction level. The backward is **issue-bound** — 3.5 instructions per
useful FMA, with address arithmetic at 27.4% of the instruction stream.

## License

[Apache License 2.0](LICENSE).
