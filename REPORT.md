# FlashAttention on 2× Tesla P100 (sm_60)

Hand-written FlashAttention-2 for Pascal — no tensor cores, no `cp.async`, no
`ldmatrix`, 48 KB shared memory per block. Every matmul is a register-blocked
`__hfma2` outer product.

Built and measured 2026-09-06 against the staged plan in `~/PLAN.md`.
All seven phases complete. Raw measurements: `BENCH.log` (append-only).

---

## Headline

**3.0–3.9× faster than `torch.nn.functional.scaled_dot_product_attention` on a
full training step, at 10% less peak memory.** Arbitrary sequence length,
d=64 or d=128, MHA / GQA / MQA.

| (1,16,N,64) | causal | torch step | this kernel | speedup | peak MB |
|---|---|---|---|---|---|
| N=2048 | no | 48.082 ms | **15.342 ms** | **3.13×** | 46 → **42** |
| N=2048 | yes | 25.122 ms | **8.068 ms** | **3.11×** | 46 → **42** |
| N=4096 | no | 190.203 ms | **57.756 ms** | **3.29×** | 93 → **84** |
| N=4096 | yes | 97.350 ms | **29.596 ms** | **3.29×** | 93 → **84** |

Split by half — **the backward is the bigger win, not the forward**:

| | forward | backward | full step |
|---|---|---|---|
| N=4096 causal | 2.15× | **3.46×** | 3.29× |

At **d=128 it is better still** — and this overturns what Phase 3 concluded:

| (1,8,N,128) | causal | torch step | this kernel | speedup |
|---|---|---|---|---|
| N=2048 | yes | 45.339 ms | **12.043 ms** | **3.76×** |
| N=4096 | no | 344.995 ms | **93.198 ms** | **3.70×** |
| N=4096 | yes | 176.317 ms | **45.485 ms** | **3.88×** |

Baseline is torch's `mem_efficient` backend, which is what SDPA actually selects
on sm_60 (`flash` reports "No available kernel"). This is a competent baseline,
not a strawman.

---

## Phase results

| phase | gate | result |
|---|---|---|
| 0 hfma2 | ratio ≈ 2× | **1.82×** — fp16 15.79 TFLOPS, fp32 8.68 ✅ |
| 1 GEMM | ≥50% of cuBLAS | **93.1%** — 14.10 TFLOPS at 4096³ ✅ |
| 2 forward d=64 | 4 regimes + beat unfused | all 4 clean w/ fp32 MACs, 1.92× ✅ |
| 3 causal / d=128 | — | causal **3.15×**; d=128 at parity ✅ |
| 4 backward | 4 regimes | all 4 clean, 3.16 TFLOPS ✅ |
| 5 torch binding | sm_60 in arch list | **yes**, no source build needed ✅ |
| 6 two GPUs | — | **1.99× of ideal 2.00×** ✅ |

### Hardware baseline (measured, not spec)

| | spec | measured |
|---|---|---|
| fp16 packed half2 | 19.04 TFLOPS | **15.79** (83%) |
| fp32 FMA | 9.52 TFLOPS | **8.68** (91%) |
| HBM2 bandwidth | 732 GB/s | **498.8** (68%) |
| clock under load | 1328 MHz | **1328, no throttle**, 140 W of a 175 W cap |

CUDA **12.4** (not 12.6 — do not "upgrade", CUDA 13 dropped Pascal).
Default **gcc-15 compiles CUDA fine**; the g++-13 requirement in `CLAUDE.md` is
a llama.cpp constraint only.

---

## Where the plan was wrong

This is the part worth re-reading. Seven corrections, all measured.

### 1. The break-even math was miscalculated — in the pessimistic direction

The plan compared **spec** bandwidth (732 GB/s) against **achieved** compute.
Apples to oranges. With measured bandwidth (498.8 GB/s), break-even
`C = d·B/2` is **16.0 TFLOPS at d=64**, not 23.4.

The backward then confirmed the corrected formula exactly: predicted a 5.1×
recompute penalty, measured 5.1× (10.9 ms to recompute S vs 2.15 ms to store
and reload it). Recomputation is still a time loss — and still correct, because
memory is the point.

### 2. The d=128 precision bug was not the accumulation length

The plan says to suspect the 128-term fp16 sum. Wrong suspect; chasing it would
have cost a day. The culprit was **pre-scaling Q by `1/√d` in fp16**. At d=64
that constant is 0.125 — a power of two, exact. At d=128 it is 0.0884, which
rounds *every Q entry* before any MAC, which is why the error survived even
exact fp32 products.

**Fix: don't pre-scale Q. Apply `1/√d` to the fp32 logit afterwards.** Free, and
took d=128 regime 2 from **9.6e-2 to 8.3e-4**.

### 3. Regime 2's 5e-2 threshold is unreachable by any fp16 kernel

Before concluding your kernel is broken, run the *unfused cuBLAS path* against
the same fp64 reference. It scores **5.918e-01** where the naive fused kernel
scores **5.915e-01**. Identical. A regime-2 failure is a statement about fp16,
not about your kernel.

Judge accuracy against the path you would otherwise run.

### 4. Two occupancy claims are wrong

- "16 KB smem allows 4 blocks/SM = 50% occupancy at 128 threads" — 4 × 128 = 512
  of 2048 threads, which is **25%**.
- The fastest GEMM runs at **25% occupancy** and beats the 50% version by 8.8%.
  Register prefetch is worth more than warps on an in-order, latency-bound chip.
  **Do not trade register tile size for occupancy without measuring.**

### 5. Phase 5's blocker was the interpreter, not the arch list

The plan braces for the sm_60 wheel situation and says "plan for
build-from-source." Measured: **no source build needed** — `torch 2.4.1+cu121`
ships `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90`.

The actual blocker: Ubuntu 26.04 ships **Python 3.14 only**, and no torch older
than 2.9.0 exists for 3.14. Fix is a 3.12 alongside:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa    # does carry a `resolute` dist
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
cd ~/fa-p100 && python3.12 -m venv .venv
.venv/bin/pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 \
    --timeout 30 --retries 20        # ~5 GB: torch + the whole nvidia-* stack
```

### 6. The "saves 537 MB" claim does not hold against torch

Phases 2–3 measured against a hand-rolled unfused cuBLAS path and concluded the
fused kernel's unconditional advantage was never materializing the N² score
matrix. **torch already avoids it on Pascal.** Backend availability on sm_60:

| backend | available | peak (1,16,2048,64) |
|---|---|---|
| `flash` | **no** — "No available kernel" | — |
| `mem_efficient` | **yes** — SDPA's default here | **16.8 MB** |
| `math` | yes | 297.9 MB |

The memory claim holds against a naive or `math`-backend implementation only.
The real result — 3× faster at equal-or-better memory against a competent
baseline — is a stronger claim anyway.

### 7. "d=128 is at parity" was wrong — two measurement errors compounded

Phase 3 concluded d=128 bought memory but not speed. That was measured
**forward-only**, against the **hand-rolled unfused cuBLAS path**. Both halves
of that were the wrong choice: training pays for the backward, and nobody runs
the unfused path.

Measured properly — full training step, against torch SDPA — **d=128 is the
best case at 3.49–3.88×, beating d=64's 2.99–3.11×.** The lesson generalizes:
*benchmark the whole operation against the thing you would otherwise run.*

### 8. `Bc=16` for d=128 is worse, not better

The plan suggests dropping to `Br=64, Bc=16` for d=128. Measured: **18% slower**
than `Bc=32`. The tile sweep (Br ∈ {32,64}, Bc ∈ {16,32,64}, 128/256 threads,
24–48 KB smem) found nothing hiding — everything landed 5.9–8.5 TFLOPS.

---

## The backward: four dead ends and one real cause

The backward runs at 6.9× its own forward against a 2.5× flop ratio. Hypotheses
tested in order — **only the last one moved**:

| hypothesis | result |
|---|---|
| dQ `atomicAdd` contention | **0.1–8%** — plan's options 2 (split-K) and 3 (separate dQ pass) are unnecessary |
| MAC rate — `__hfma2` on **S/dP and dQ** | **+13%**, and destroys accuracy (regimes 2/4 → ~1e0) |
| shared-fragment count (tile 1×16 → 2×8) | **+0.2%** |
| `__half2float` count (hoist out of pairing loop) | **0%** — ptxas had already CSE'd it |
| global traffic — `BC` 32→64, halving Q/dO re-reads | **+6.8%** |
| **`__hfma2` in the dV/dK contraction only** | **+7.8% / +13.2% causal, accuracy intact** ✅ |

### Then stop guessing and read the SASS

After four dead ends the useful move was `cuobjdump -sass`, not a fifth
hypothesis. `bwd_kernel<64,64,64,128>` is 6473 instructions:

| | count | share | |
|---|---|---|---|
| `FFMA` | 1856 | 28.7% | the actual math |
| `HADD2` | 864 | **13.3%** | `__half2float` — Pascal compiles it to `HADD2.F32 Rd, Rs.H0_H0, -RZ` |
| `XMAD`/`MOV`/`IADD`/`LEA`/`SHR`/`ISETP` | 1772 | **27.4%** | address arithmetic |

**It is issue-bound — 3.5 instructions retired per useful FMA** — and
conversions alone are a seventh of the kernel. That also retires the
"traffic-bound" reading: `BC` 32→64 did help, but the instruction stream is the
real ceiling.

**The fix, and the transferable lesson.** The failed half2 experiment had put
`__hfma2` in the *outer products*, exactly where the `dS = P(dP − D)`
cancellation lives. The **dV/dK contraction** is the opposite case: `P ∈ [0,1]`
against unit-variance `dO`, only 8 terms before draining to fp32, and it is
where the conversions concentrate. Same instruction, opposite verdict, decided
entirely by *which matmul it lands in*.

Note the framing correction: Phase 4 judged the backward harshly, but **torch's
`mem_efficient` backward is 9.0× its own forward** vs this kernel's 6.9×.
Attention backward is simply bad on Pascal; this one is less bad, and it is where
most of the 3.06× comes from.

### Validation

Three layers, so a failure localizes:

1. **Finite differences vs the analytic fp64 reference: 4.16e-10.** Do this
   first — without it, later checks only prove the kernel agrees with whatever
   formula you typed. (`torch.autograd.gradcheck` is not needed and does not work
   on an fp16-only kernel.)
2. Kernel vs fp64 reference.
3. Four input regimes: N(0,1), ×10, near-uniform, near-one-hot.

All four pass, causal and non-causal, worst error 2.1e-2.

---

## Precision tiers

Two independent fp16 roundings in QK^T need different fixes:

| tier | what is fp32 | r1 | r2 ×10 | r3 | r4 1-hot | TFLOPS |
|---|---|---|---|---|---|---|
| `drain64` | nothing | 1.4e-2 | **5.9e-1** | 4.5e-3 | **8.9e-2** | 9.76 |
| `drain16` | the sum | 7.8e-3 | **1.7e-1** | 4.4e-3 | 2.3e-2 | 9.51 |
| `fp32mac` | sum **and** products | 6.0e-3 | 8.1e-4 | 4.4e-3 | 1.5e-3 | 6.83 |

Draining fixes the *sum* and cannot fix the *products*. Only `fmaf` on converted
halves clears all four regimes, at a 28% cost.

**Defaults:** `fp32mac` for unnormalised or unknown inputs; `drain16` for
LayerNorm'd O(1) activations (the normal case). **Never ship `drain64`** — 2.6%
faster than `drain16` and it fails realistic peaked attention.

The backward is fp32 throughout and deliberately has no fp16 tier:
`dS = P·(dP − D)` is a cancellation.

---

## Build traps

- **`AT_CUDA_CHECK` needs `#include <ATen/cuda/Exceptions.h>`.**
  `torch/extension.h` does not pull it in.
- **`pip install ninja` is not enough.** `cpp_extension` shells out to the ninja
  *binary*, and `.venv/bin` is only on `PATH` if the venv is **activated** —
  running `.venv/bin/python` directly does not put it there. `import ninja`
  succeeding proves nothing. Same family as the `-ts` comma-vs-slash trap in
  `CLAUDE.md`: the check you naturally run is not the check the tool performs.
- **`llama-bench -ts` uses `/`, `llama-server` uses `,`** — unrelated to this
  project but the same trap family, already in `CLAUDE.md`.
- **Non-issue, contrary to expectation:** system nvcc **12.4** against torch's
  bundled CUDA **12.1** headers produced no conflict.
- **`/tmp` is tmpfs, 46 GB, RAM-backed.** A 5 GB pip download stages entirely
  through RAM on a box that also runs 70–80 GB-resident models.

---

## What it supports

| | |
|---|---|
| head dim | **64 or 128** |
| sequence length | **arbitrary** — ragged tails masked, no padding required |
| attention | MHA, **GQA, MQA** (`Hq` any multiple of `Hkv`) |
| causal | yes, and it skips the upper triangle rather than masking it |
| precision | fp16 storage; fp16 or fp32 MACs in the forward, fp32 backward |
| dtype | fp16 in, fp16 out, fp16 gradients (`dQ` accumulates in fp32 internally) |

Ragged-`N` support is a compile-time template parameter, not a runtime branch:
the launcher picks the aligned instantiation when `N % BR == 0 && N % BC == 0`.
Written as a runtime check it cost **7% on the backward**; this way it costs
nothing on aligned shapes.

## Remaining limits

- **Backward is global-memory-traffic bound** and runs at 6.9× its own forward.
  It still beats torch's by 3.2×, but the headroom is real — see above.
- **No dropout, no attention bias/ALiBi, no arbitrary mask** — causal or dense
  only.
- **No sliding-window or block-sparse patterns.**
- **fp16 only** — bf16 has no hardware support on Pascal at all.
- Head dims other than 64 and 128 are not instantiated.

---

## Files

| file | phase |
|---|---|
| `common.cuh` | harness — timing, fp64 references, error metrics |
| `bench_cublas.cu` | 1.1 — cuBLAS baselines on attention shapes |
| `gemm.cu` | 1 — half2 SIMT GEMM, four variants |
| `flash_fwd.cuh` + `attention.cu` | 2, 3 — forward, causal, d=128 |
| `flash_bwd.cuh` + `bwd.cu` | 4 — backward, three-layer validation |
| `fa_p100_ext.cu` + `fa_p100.py` | 5 — torch binding + `autograd.Function` |
| `test_fa_p100.py` | regression suite — 383 checks across both head dims, MHA/GQA/MQA, ragged N, four input regimes |
| `bench_train.py` | full training step vs torch SDPA |
| `multigpu.cu` | 6 — two-GPU data-parallel scaling |

## Running it

```bash
cd ~/fa-p100
make                                    # all standalone CUDA binaries

# regression suite: 383 checks, exits non-zero on failure
env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python test_fa_p100.py
numactl --cpunodebind=0 --membind=0 ./attention      # pin to card 0's NUMA node

# from Python — .venv/bin MUST be on PATH or cpp_extension cannot find ninja
env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python bench_train.py
```

```python
from fa_p100 import flash_attn

# q: (B, Hq, N, D)   k, v: (B, Hkv, N, D)   D in {64, 128}, N arbitrary
# Hkv == Hq -> MHA;  Hkv == 1 -> MQA;  otherwise GQA
o = flash_attn(q, k, v, causal=True)
o = flash_attn(q, k, v, causal=True, full_precision=True)   # fp32 MACs, +28%
```

**Benchmark on the serving config or the numbers are meaningless**, and pin
NUMA — card 0 is node 0, card 1 is node 1 on the CPU2 riser.
