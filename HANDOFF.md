# Handoff — FlashAttention on 2× Tesla P100

**Read this before touching anything.** `REPORT.md` is what was built and
measured. This document is what to do next and how not to waste days
re-deriving things that were already settled.

Repo: `~/fa-p100` (git, 4 commits). Plan of record: `~/PLAN.md` — **partly
superseded, see "Traps in the documentation" below.** Measurements:
`BENCH.log`, append-only.

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
| HBM2 | 732 GB/s | **498.8** |
| clock under load | — | **1328 MHz, no throttle, 140 W of a 175 W cap** |

The 175 W cap **does not bind** for compute-bound fp16 work. CUDA is **12.4**
and must stay there (CUDA 13 dropped Pascal). Default **gcc-15 compiles CUDA
fine** — the g++-13 note in `~/CLAUDE.md` is a llama.cpp constraint only.

## 4. Traps in the documentation itself

`~/PLAN.md` is the original plan, annotated with results. **Several of its
conclusions were later overturned by better measurement**, and the phase
sections still contain the superseded reasoning alongside corrections.
`REPORT.md` is authoritative where they disagree. Specifically wrong in the
original plan:

1. Break-even math compared **spec** bandwidth (732) against **achieved**
   compute. With measured bandwidth the d=64 threshold is 16.0 TFLOPS, not 23.4.
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

## 5. The goal, and what to do next

**The original goal is met.** The plan's seven phases are complete and the
kernel beats the best thing torch offers on this hardware by 3–4×.

**The next objective has not been chosen — that is a decision for the user, not
an assumption for you to make.** The three plausible directions, with honest
assessments:

### A. Use it in something real *(recommended if the point was ever to use it)*
Nothing currently consumes these kernels. They are a validated library with no
caller. Wiring them into an actual training run is the only way to find the
integration bugs a synthetic test suite cannot. Note the box's serving stack is
**llama.cpp/llama-swap, which is C++ and does not use PyTorch** — so a PyTorch
training job is a different workload from what this machine currently runs.

### B. Close the remaining feature gaps
No dropout, no attention bias / ALiBi, no arbitrary mask, no sliding-window or
block-sparse, head dims restricted to 64 and 128, fp16 only (bf16 is impossible
— Pascal has no hardware for it). Each is self-contained. Attention bias and
arbitrary masks are the ones real models most often need.

### C. Keep optimising the backward
It runs at 6.5× its own forward against a 2.5× flop ratio. It is **issue-bound**
(see §3), so the levers are: reduce address arithmetic in the three inner loops,
or shrink the six-tile 48 KB shared footprint (`Qs, dOs, Ks, Vs, Ps, dSs`) so
`BR` can grow and Q/dO get re-read less. **This is a redesign, not a tweak**,
and it improves a number that is already winning by 3.4×. Lowest value of the
three.

### Small and cheap, whoever picks this up
- **`attention.cu` benchmarks the wrong baseline.** It still prints
  `0.58x unfused <-- slower` for d=128 against the hand-rolled cuBLAS path.
  Correct code, misleading framing, and the last place in the repo still
  quoting a comparison that was shown to be wrong. Point it at torch SDPA.
- `~/.cache/pip` holds 3.0 GB that is reclaimable.
- A pending kernel upgrade (`7.0.0-31` in `/boot`, running `-30`) — **the user
  has explicitly deferred this.** Do not reboot; it takes llama-swap down.

## 6. Layout

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

**Precision tiers.** Forward: `drain16` for LayerNorm'd O(1) activations,
`fp32mac` (+28%) when inputs are unnormalised or unknown. Never ship
`drain64` — 2.6% faster than `drain16` and it fails realistic peaked attention.
Backward is fp32 in the outer products and half2 in the dV/dK contraction; that
split is deliberate and measured, do not "simplify" it.

**Validation is three-layer and the order matters.** Finite differences against
the analytic fp64 reference *first* (currently 4.16e-10) — without it, every
later check only proves the kernel agrees with whatever formula was typed into
the reference, not with calculus.
