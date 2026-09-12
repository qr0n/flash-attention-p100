# Handoff — FlashAttention on 2× Tesla P100

**Read this before touching anything.** `REPORT.md` is what was built and
measured. This document is what to do next and how not to waste days
re-deriving things that were already settled.

Repo: `~/fa-p100` (git). Plan of record: `~/PLAN.md` — **partly superseded,
see "Traps in the documentation" below.** Measurements: `BENCH.log`,
append-only.

---

## 1. What this is

A from-scratch FlashAttention-2 forward and backward for **Tesla P100
(GP100, sm_60)**. Pascal has no tensor cores, no `cp.async`, no `ldmatrix`, and
a hard 48 KB shared-memory-per-block cap. Every matmul is a register-blocked
`__hfma2` outer product written by hand.

Two cards, `3b:00.0` (NUMA node 0) and `af:00.0` (NUMA node 1, CPU2 riser),
capped to 175 W each. The box is also the household LLM host — llama-swap runs
on `:8080` — so **check `nvidia-smi` for a loaded model before benchmarking**,
or your numbers are garbage and you may evict someone's model.

## 2. Current state — it works and it is fast

**3.29× faster than `torch.nn.functional.scaled_dot_product_attention` on a
full training step at d=64, 3.92× at d=128**, at ~10% less peak memory.

| (1,H,N,D) | causal | torch | flash | speedup |
|---|---|---|---|---|
| 1,16,2048,64 | yes | 25.11 ms | 8.05 ms | **3.12×** |
| 1,16,4096,64 | yes | 97.26 ms | 29.57 ms | **3.29×** |
| 1,8,4096,128 | yes | 176.11 ms | 44.92 ms | **3.92×** |

Supported: head dim **64 or 128**, **arbitrary sequence length**, **MHA / GQA /
MQA**, causal or dense, fp16 in/out with fp32 gradient accumulation internally.

**There is now a real caller.** The kernels are registered as a HuggingFace
attention implementation and run a stock, unmodified Qwen3-0.6B: see
"Using it from transformers" below. That was section 5A of this document and
it is done.

`test_fa_p100.py` is 383 checks and exits non-zero on failure. **Run it before
and after every kernel change.** It takes a few minutes.

```bash
cd ~/fa-p100 && make
env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python test_fa_p100.py
```

The `PATH` prefix is not optional — `torch.utils.cpp_extension` shells out to
the **ninja binary**, and `.venv/bin` only lands on `PATH` when the venv is
*activated*. Running `.venv/bin/python` directly is not enough. `import ninja`
succeeding proves nothing.

## 3. Do not repeat these — all measured, all dead ends

### Backward optimisation attempts that failed

| tried | result |
|---|---|
| dQ `atomicAdd` contention (assumed to be the bottleneck) | **0.1–8%.** Split-K and a separate dQ pass are both unnecessary |
| `__hfma2` in the **outer products** (S/dP, dQ) | **+13% but wrecks accuracy** — regimes 2/4 blow out to ~1e0 |
| dV/dK shared-fragment count, tile 1×16 → 2×8 | **+0.2%** |
| hoisting `__half2float` out of the pairing loop | **0%** — ptxas had already done it |

What *did* work: `BC` 32→64 (+6.8%), and `__hfma2` in the **dV/dK contraction
only** (+7.8% / +13.2% causal, accuracy intact).

**The transferable lesson:** the same instruction is ruinous in the outer
products and safe in the contraction. `dS = P(dP − D)` is a cancellation;
`dV = Pᵀ dO` is `P ∈ [0,1]` against unit-variance `dO` over only 8 terms before
draining to fp32. Decide per matmul, not per kernel.

### When you run out of hypotheses, read the SASS

```bash
cuobjdump -sass bwd | grep -A99999 "bwd_kernel.*Li64ELi64ELi64ELi128E"
```

Doing this after four dead ends is what actually found the answer.
`bwd_kernel<64,64,64,128>` is 6473 instructions: **FFMA 28.7%**, **`HADD2`
13.3%** (that is `__half2float` — Pascal compiles it to
`HADD2.F32 Rd, Rs.H0_H0, -RZ`), **address arithmetic 27.4%**. It is
**issue-bound at 3.5 instructions per useful FMA**, not bandwidth-bound.

### Measurement discipline — this cost two wrong conclusions

**Benchmark the full training step against `torch.nn.F.scaled_dot_product_attention`.**
Not forward-only, and not against a hand-rolled unfused cuBLAS path.

Both errors were made here and both produced wrong conclusions that survived
into the plan:

- "d=128 is at parity" — measured forward-only vs unfused cuBLAS. It is
  actually the **best** case (3.92×).
- "the backward is the weak half" — it is the **strong** half (3.46× vs the
  forward's 2.15×). torch's own backward is 9.0× its forward; this one is 6.5×.

On sm_60 torch's SDPA selects the **`mem_efficient`** backend (`flash` reports
"No available kernel", `math` costs 297.9 MB vs 16.8). It already avoids the
N² score matrix, so "we save a 537 MB score matrix" is **only true against a
naive baseline**, not against torch.

### Hardware facts, measured — do not re-derive

| | spec | measured |
|---|---|---|
| fp16 packed half2 | 19.04 TFLOPS | **15.79** |
| fp32 FMA | 9.52 TFLOPS | **8.68** |
| HBM2, read+write (hardest pattern) | 732 GB/s | **498.8** |
| HBM2, pure read (the real ceiling) | 732 GB/s | **607** |
| clock under load | — | **1328 MHz, no throttle, 140 W of a 175 W cap** |

The 175 W cap **does not bind** for compute-bound fp16 work. CUDA is **12.4**
and must stay there (CUDA 13 dropped Pascal). Default **gcc-15 compiles CUDA
fine** — the g++-13 note in `~/CLAUDE.md` is a llama.cpp constraint only.

## 4. Using it from transformers — the real caller

```python
from transformers import AutoModelForCausalLM
from fa_p100_hf import enable_fa_p100

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).cuda()
enable_fa_p100(model)      # registers `fa_p100` and switches the model onto it
```

Verified on stock **Qwen3-0.6B** (28 layers, 16/8 GQA, head_dim 128 — the
kernel's best case) with **transformers 4.57.6**. The model source is
untouched; `fa_p100_hf.py` only registers an entry in `ALL_ATTENTION_FUNCTIONS`.

**transformers is pinned to 4.x on purpose.** 5.x requires torch>=2.5, and
torch is pinned to 2.4.1 because it is the last build shipping `sm_60`. Do not
"upgrade" either one — same trap family as CUDA 13 dropping Pascal.

### Correctness: measured against the eager-vs-sdpa envelope

`verify_hf.py` runs the same weights through sdpa, eager, and fa_p100. The
question is not "is fa_p100 close to sdpa" in the abstract — it is whether
fa_p100 sits inside the envelope *two torch backends already occupy* when they
disagree with each other from summation order alone.

| seq_len | mean | p99 | p99.99 | loss gap | top-1 vs sdpa |
|---|---|---|---|---|---|
| 1024 | 1.01x | 1.00x | 0.48x | −3.02e-4 nats | 99.41% (eager 99.71%) |
| 2048 | 1.03x | 1.00x | 1.99x | −2.43e-4 nats | 99.41% (eager 99.61%) |

**Do not gate on max-abs.** Over 2048x151936 = 311M logits the maximum is an
extreme order statistic. It swung **0.48x → 3.89x** of eager between seq_len
1024 and 2048 *on an unchanged kernel*, and at 1024 it is **eager** that has
1202 logits over 0.2 while fa_p100 has zero. An earlier version of
`verify_hf.py` gated on it and produced a spurious FAIL. Gate on mean, p99.99,
loss and top-1 agreement, as it does now.

**Error does not grow with sequence position** — mean deviation per 256-token
bucket is flat (6.7e-3 → 6.4e-3), the same shape as eager's. That was the
hypothesis worth testing, since online softmax rescales over more blocks as N
grows. It is not happening.

### Speed: 3.9x in the microbenchmark is 1.15–1.62x end to end

`finetune_hf.py` runs the *same* fine-tune twice from the same initial weights
over the same data in the same order, once per implementation.

| seq_len | sdpa | fa_p100 | speedup | loss drift |
|---|---|---|---|---|
| 1024 | 1217.7 ms | 1061.6 ms | **1.15x** | 4.9e-4 |
| 2048 | 2934.9 ms | 2288.9 ms | **1.28x** | 3.0e-4 |
| 4096 (grad ckpt) | 9486.6 ms | 6665.6 ms | **1.42x** | 1.7e-4 |
| 8192 (grad ckpt) | 29366.0 ms | 18117.8 ms | **1.62x** | 2.5e-4 |

This is Amdahl, and it is the honest number. A 0.6B model with a **151936**
vocabulary spends most of its step in the LM head and the MLPs; attention is a
minority of the work until N gets large, which is exactly why the speedup
climbs with sequence length. Expect more on a model with a smaller
vocab-to-depth ratio, and less on a shallower one.

**Peak memory is identical (1.00x), and that is the correct result, not a
null one.** torch selects `mem_efficient` on sm_60, which already avoids the
N² score matrix, and the peak here is dominated by ~9.5 GiB of AdamW state
(fp32 weights + grads + m + v) that no attention kernel touches.

### What raises, and why none of it degrades quietly

Every unsupported path raises `NotImplementedError`. This is deliberate: the
failure mode to avoid is a model that trains to a slightly wrong answer.

- **padding / arbitrary masks.** Pack sequences to a fixed length instead.
  This one needs care: `masking_utils._preprocess_mask_arguments` early-exits
  with a **None** mask for any `_attn_implementation` absent from
  `ALL_MASK_ATTENTION_FUNCTIONS`, which is what we want (no B×H×N×N mask is
  ever built) — but it means a padding mask handed to `model()` would be
  dropped **silently**. `enable_fa_p100` installs a forward guard that catches
  it. Do not "fix" this by registering a mask function: the kernel takes no
  mask argument, so the mask would be built and then ignored.
- **KV-cache decoding.** The kernel requires `q_len == kv_len`. Caught in the
  forward guard, not the attention interface, because there it would only fire
  on the first *decode* step — i.e. after a successful prefill, halfway
  through a `generate()` call. `enable_fa_p100` sets `use_cache=False`, and
  cache-free generation works (quadratic, but correct).
- sliding-window attention, attention dropout, `output_attentions`,
  `head_mask`, head_dim ∉ {64,128}, and any scale other than `1/sqrt(D)`
  (it is **hardcoded** in the `.cu` dispatch, so a model wanting another one
  would be silently mis-scaled).

### Two traps that cost real time here

**`labels=` OOMs at seq_len 2048 on a 16 GiB card.** transformers'
`ForCausalLMLoss` upcasts the whole logit tensor to fp32 — 2048×151936×4 =
1.24 GiB, plus as much again inside `cross_entropy`. It looks like an
attention memory problem and is not; it hits sdpa identically.
`finetune_hf.py` checkpoints the LM head per chunk instead, which is what
makes seq_len ≥ 2048 reachable and therefore what makes the kernel's
contribution measurable at all.

**q and k arrive as fp32 even under `torch.autocast`.** Qwen3's per-head
`q_norm`/`k_norm` multiply by an fp32 master weight, which promotes them back
after the fp16 `q_proj`; the rotary `cos`/`sin` are fp32 for the same reason.
`v` arrives fp16. torch's SDPA never shows this because autocast intercepts
SDPA itself and casts its inputs down — **a custom attention interface has to
do it explicitly.** Getting it wrong is silent: it just runs slower.

## 5. Traps in the documentation itself

`~/PLAN.md` is the original plan, annotated with results. **Several of its
conclusions were later overturned by better measurement**, and the phase
sections still contain the superseded reasoning alongside corrections.
`REPORT.md` is authoritative where they disagree. Specifically wrong in the
original plan:

1. Break-even math compared **spec** bandwidth (732) against **achieved**
   compute. With measured bandwidth the d=64 threshold is far under the 23.4
   TFLOPS assumed. **Use 607 GB/s, not 498.8** (corrected 2026-09-12): 498.8 is
   the read-modify-write pattern, 607 is the pure-read ceiling, and the forward
   is read-dominated. That makes the d=64 threshold **19.4 TFLOPS**, not 16.0 —
   so against 15.79 measured the kernel is **19% short of memory-bound**, which
   agrees with the SASS finding above rather than contradicting it. See
   `REPORT.md` §1.
2. The d=128 precision bug is **not** the accumulation length it tells you to
   suspect — it was pre-scaling Q by `1/√d` in fp16 (`1/√128` is not a power of
   two, so it rounds every Q entry *before* any MAC). Apply the scale to the
   fp32 logit instead.
3. "4 blocks × 128 threads = 50% occupancy" is arithmetically **25%**. And the
   fastest GEMM here runs at 25% occupancy — do not trade register tile size
   for occupancy without measuring.
4. "Plan for build-from-source" for PyTorch — **not needed**, `torch 2.4.1+cu121`
   ships `sm_60`. The real blocker was Ubuntu 26.04 shipping Python 3.14 only.
5. `Bc=16` for d=128 is **18% worse** than `Bc=32`, not better.

## 6. The goal, and what to do next

**The original goal is met, and the kernels now have a real caller.** The
plan's seven phases are complete, the kernel beats the best thing torch offers
on this hardware by 3–4× in isolation, and it trains a stock HuggingFace
Qwen3-0.6B correctly at 1.15–1.62× end to end (§4).

**The next objective is a decision for the user, not an assumption for you to
make.** Assessments, updated now that A is done:

### A. Use it in something real — *done, and what it revealed*
`fa_p100_hf.py` + `verify_hf.py` + `finetune_hf.py`. What the integration
found that no synthetic suite could: the silent-mask early-exit, the
prefill-succeeds-then-decode-fails cache trap, the fp32 q/k promotion through
`q_norm` under autocast, and the fact that the LM head — not attention — is
what OOMs a 0.6B model on a 16 GiB card. All four are written up in §4.

The remaining honest gap is that this is a **demonstration fine-tune on a
small corpus**, not a production training job. If there is a model someone
actually wants trained here, that is the natural next step, and the plumbing
is now in place.

### B. Close the remaining feature gaps *(now the highest-value direction)*
No dropout, no attention bias / ALiBi, no arbitrary mask, no sliding-window or
block-sparse, head dims restricted to 64 and 128, fp16 only (bf16 is
impossible — Pascal has no hardware for it). Each is self-contained.

§4 sharpened the priority: **arbitrary masks are the binding constraint**, not
a nice-to-have. Without them the kernel cannot do padded batches (so batching
means packing), cannot do sliding-window models at all, and cannot serve a KV
cache. `attention bias` is second. Head-dim coverage is third — a good many
models are neither 64 nor 128.

### C. Keep optimising the backward
It runs at 6.5× its own forward against a 2.5× flop ratio. It is **issue-bound**
(see §3), so the levers are: reduce address arithmetic in the three inner loops,
or shrink the six-tile 48 KB shared footprint (`Qs, dOs, Ks, Vs, Ps, dSs`) so
`BR` can grow and Q/dO get re-read less. **This is a redesign, not a tweak**,
and §4 lowered its value further: at the end-to-end level attention is a
minority of the step, so a 10% kernel win is worth ~2–4% of a training run.
Still lowest value of the three.

### Small and cheap, whoever picks this up
- **`attention.cu` benchmarks the wrong baseline.** It still prints
  `0.58x unfused <-- slower` for d=128 against the hand-rolled cuBLAS path.
  Correct code, misleading framing, and the last place in the repo still
  quoting a comparison that was shown to be wrong. Point it at torch SDPA.
- `~/.cache/pip` holds 3.0 GB that is reclaimable. The venv now also carries
  transformers and a 1.5 GB Qwen3-0.6B in `~/.cache/huggingface`.
- A pending kernel upgrade (`7.0.0-31` in `/boot`, running `-30`) — **the user
  has explicitly deferred this.** Do not reboot; it takes llama-swap down.

## 7. Layout

| file | what |
|---|---|
| `flash_fwd.cuh` | forward kernel — templated on `BR, BC, D, THREADS, CAUSAL, DRAIN, FP32MAC, BOUNDS` |
| `flash_bwd.cuh` | backward — adds `NO_ATOMIC` (diagnostic), `QDRAIN`, `GOutT`, `H2_DVDK` |
| `fa_p100_ext.cu` / `fa_p100.py` | torch binding + `autograd.Function` |
| `test_fa_p100.py` | **383-check regression suite, run it every time** |
| `bench_train.py` | full training step vs torch SDPA — the benchmark that matters |
| `common.cuh` | timing, fp64 references, error metrics |
| `gemm.cu`, `bench_cublas.cu` | Phase 1 GEMM and cuBLAS baselines |
| `attention.cu`, `bwd.cu` | standalone CUDA drivers (see the caveat above) |
| `multigpu.cu` | two-GPU scaling — 1.99× of ideal, nothing to do here |
| `fa_p100_hf.py` | **HuggingFace attention implementation + guards (§4)** |
| `verify_hf.py` | parity vs sdpa/eager on a real model — exits non-zero on failure |
| `finetune_hf.py` | the same fine-tune on both backends, loss curves + wall clock |

**Precision tiers.** Forward: `drain16` for LayerNorm'd O(1) activations,
`fp32mac` (+28%) when inputs are unnormalised or unknown. Never ship
`drain64` — 2.6% faster than `drain16` and it fails realistic peaked attention.
Backward is fp32 in the outer products and half2 in the dV/dK contraction; that
split is deliberate and measured, do not "simplify" it.

**Validation is three-layer and the order matters.** Finite differences against
the analytic fp64 reference *first* (currently 4.16e-10) — without it, every
later check only proves the kernel agrees with whatever formula was typed into
the reference, not with calculus.
