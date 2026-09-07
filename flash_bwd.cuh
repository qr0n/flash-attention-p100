// Phase 4 — FlashAttention-2 backward pass for GP100 (sm_60).
//
// FA2 inverts the forward's loop nest: the OUTER loop is over K/V blocks, the
// inner over Q blocks. That makes dK_j and dV_j block-local -- they live in
// registers across the whole inner loop and are written exactly once. dQ_i is
// the awkward one: it collects a contribution from every (i, j) pair, so it
// needs cross-block accumulation. This file implements the plan's option 1,
// fp32 atomicAdd into global dQ, which is the one to measure before assuming
// anything fancier is needed.
//
// Per (i, j) block, with L = logsumexp and Dv = rowsum(dO * O) both precomputed:
//
//   S  = Q_i K_j^T * scale        P  = exp(S - L_i)
//   dV_j += P^T dO_i              dP = dO_i V_j^T
//   dS = P * (dP - Dv_i)
//   dQ_i += dS K_j * scale        dK_j += dS^T Q_i * scale
//
// Five matmuls with three different contraction axes. Every shared tile is
// stored with the contraction axis as the SLOW index and the free axis
// contiguous, so all three structures read 16-byte fragments:
//
//   Qs, dOs : [k][r]   Ks, Vs : [k][c]   Ps, dSs : [c][r]
//
//   S, dP  contract over k -> outer product, thread owns 8 r x TC c
//   dV, dK contract over r -> dot product,   thread owns 1 c x TK k
//   dQ     contract over c -> outer product, thread owns 8 r x TKQ k

#pragma once
#include <type_traits>
#include "common.cuh"

namespace flash_bwd {

constexpr int TR = 8;

template <int BR, int BC, int D, int THREADS>
struct Cfg {
    static constexpr int ROWG = BR / TR;
    static constexpr int TXR  = THREADS / ROWG;
    static constexpr int TC   = BC / TXR;          // S/dP columns per thread
    static constexpr int TKQ  = D / TXR;           // dQ columns per thread
    // dV/dK tile. The naive choice -- one K/V row per thread and every head
    // column -- is maximally lopsided and reads 34 fragments to do 256 MACs.
    // Holding TCV rows and streaming TKV columns cuts that to 20 fragments for
    // the same MACs, because the P/dS fragments are reused across TKV columns.
    static constexpr int TCV   = 2;
    static constexpr int CGRPV = BC / TCV;          // c-groups
    static constexpr int TPCG  = THREADS / CGRPV;   // threads per c-group
    static constexpr int TKV   = D / TPCG;          // dK/dV columns per thread
    static constexpr int QVEC = BR * D / 8 / THREADS;
    static constexpr int KVEC = BC * D / 8 / THREADS;
    static constexpr int SMEM = (2 * D * BR + 2 * D * BC + 2 * BC * BR) * 2;

    static_assert(BR % TR == 0 && BR % BC == 0, "BR multiple of 8 and of BC");
    static_assert(BC % TXR == 0 && TC >= 1, "BC too small for this BR/THREADS");
    static_assert(D % TXR == 0, "head dim must split evenly");
    static_assert(BC % TCV == 0 && THREADS % CGRPV == 0 && D % TPCG == 0,
                  "dV/dK tile does not divide evenly");
    static_assert(TXR <= 32, "row reduction must stay inside a warp");
    static_assert(SMEM <= 49152, "over the 48 KB per-block shared memory cap");
};

// Dv_i = rowsum(dO_i * O_i). One block per row; this is cheap and bandwidth
// bound, so it stays a separate kernel rather than being recomputed per tile.
template <int D>
__global__ void bwd_preprocess(const __half* __restrict__ O,
                               const __half* __restrict__ dO,
                               float* __restrict__ Dv, int N) {
    const size_t row = (size_t)blockIdx.y * N + blockIdx.x;
    const __half* o = O + row * D;
    const __half* d = dO + row * D;
    float acc = 0.f;
    for (int k = threadIdx.x; k < D; k += blockDim.x)
        acc += __half2float(o[k]) * __half2float(d[k]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (threadIdx.x == 0) Dv[row] = acc;
}

// NO_ATOMIC replaces the dQ atomicAdd with a plain store. It produces WRONG
// gradients and exists only to price the atomics: the difference between the
// two builds is what contention plus read-modify-write traffic is costing.
// QDRAIN selects the arithmetic for the two outer-product matmuls (S/dP and
// dQ). 0 keeps them in fp32 fmaf, which is what makes the reference build clear
// every input regime. Any positive value switches to __hfma2 with an fp32 fold
// every QDRAIN steps -- the same tier structure that the forward pass uses, and
// with the same trade: roughly 2x the MAC rate for fp16 product rounding.
// The dV/dK contraction stays fp32 in both; it reduces over BR and is the
// numerically touchiest of the five.
// GOutT is the storage type for dK and dV. They are accumulated in fp32 and
// written exactly once from registers, so there is no reason to stage them
// through an fp32 buffer just to cast afterwards -- that doubles their
// footprint at peak. dQ has no such choice: it is built by atomicAdd across
// blocks and must stay fp32.
template <int BR, int BC, int D, int THREADS, bool CAUSAL, bool NO_ATOMIC = false,
          int QDRAIN = 0, typename GOutT = float, bool BOUNDS = true,
          bool H2_DVDK = false>
__global__ __launch_bounds__(THREADS) void bwd_kernel(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, const __half* __restrict__ dO,
    const float* __restrict__ Lse, const float* __restrict__ Dv,
    float* __restrict__ dQ, GOutT* __restrict__ dK, GOutT* __restrict__ dV,
    int N, float scale, int Hq, int GROUP) {

    using C = Cfg<BR, BC, D, THREADS>;
    __shared__ __align__(16) __half Qs [D * BR];    // [k][r]
    __shared__ __align__(16) __half dOs[D * BR];    // [k][r]
    __shared__ __align__(16) __half Ks [D * BC];    // [k][c]
    __shared__ __align__(16) __half Vs [D * BC];    // [k][c]
    __shared__ __align__(16) __half Ps [BC * BR];   // [c][r]
    __shared__ __align__(16) __half dSs[BC * BR];   // [c][r]

    const int tid = threadIdx.x;
    const int ty  = tid / C::TXR;          // row group, for the (r,*) mappings
    const int tx  = tid % C::TXR;
    const int cb  = (tid / C::TPCG) * C::TCV;   // this thread's K/V rows
    const int kb0 = (tid % C::TPCG) * C::TKV;   // and its head columns
    const int j0  = blockIdx.x * BC;

    // Grouped-query attention inverts awkwardly here: one K/V head is shared by
    // GROUP query heads, so dK and dV must accumulate over all of them. Making
    // blockIdx.y index the K/V head and looping the group INSIDE keeps that
    // accumulation in registers -- the alternative is atomics on dK/dV.
    const int Hkv   = Hq / GROUP;
    const int kv_bh = blockIdx.y;                 // over B*Hkv
    const int bidx  = kv_bh / Hkv;
    const int kvh   = kv_bh % Hkv;
    K  += (size_t)kv_bh * N * D;
    V  += (size_t)kv_bh * N * D;
    dK += (size_t)kv_bh * N * D;
    dV += (size_t)kv_bh * N * D;

    // ---- stage K and V once; they are resident for this block's whole life --
#pragma unroll
    for (int v = 0; v < C::KVEC; ++v) {
        const int flat = v * THREADS + tid;
        const int c = flat / (D / 8), kb = (flat % (D / 8)) * 8;
        const int gc = j0 + c;
        const float4 rk = (!BOUNDS || gc < N) ? *reinterpret_cast<const float4*>(&K[(size_t)gc * D + kb])
                                   : make_float4(0.f, 0.f, 0.f, 0.f);
        const float4 rv = (!BOUNDS || gc < N) ? *reinterpret_cast<const float4*>(&V[(size_t)gc * D + kb])
                                   : make_float4(0.f, 0.f, 0.f, 0.f);
        const __half* hk = reinterpret_cast<const __half*>(&rk);
        const __half* hv = reinterpret_cast<const __half*>(&rv);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            Ks[(kb + i) * BC + c] = hk[i];
            Vs[(kb + i) * BC + c] = hv[i];
        }
    }

    float dv_acc[C::TCV][C::TKV], dk_acc[C::TCV][C::TKV];
#pragma unroll
    for (int c = 0; c < C::TCV; ++c)
#pragma unroll
        for (int t = 0; t < C::TKV; ++t) { dv_acc[c][t] = 0.f; dk_acc[c][t] = 0.f; }

    // Causal: K block j is invisible to every Q row below j0, so start at the
    // first Q block that reaches the diagonal. BR is a multiple of BC, so this
    // lands on a block boundary.
    const int i_start = CAUSAL ? (j0 / BR) * BR : 0;

    for (int g = 0; g < GROUP; ++g) {
      const int    qh   = bidx * Hq + kvh * GROUP + g;
      const size_t qoff = (size_t)qh * N * D;
      const __half* __restrict__ Qg  = Q  + qoff;
      const __half* __restrict__ dOg = dO + qoff;
      float* __restrict__        dQg = dQ + qoff;
      const float* __restrict__  Lseg = Lse + (size_t)qh * N;
      const float* __restrict__  Dvg  = Dv  + (size_t)qh * N;

      for (int i0 = i_start; i0 < N; i0 += BR) {
        __syncthreads();
#pragma unroll
        for (int v = 0; v < C::QVEC; ++v) {
            const int flat = v * THREADS + tid;
            const int r = flat / (D / 8), kb = (flat % (D / 8)) * 8;
            const int gr = i0 + r;
            const float4 rq = (!BOUNDS || gr < N) ? *reinterpret_cast<const float4*>(&Qg[(size_t)gr * D + kb])
                                       : make_float4(0.f, 0.f, 0.f, 0.f);
            const float4 rd = (!BOUNDS || gr < N) ? *reinterpret_cast<const float4*>(&dOg[(size_t)gr * D + kb])
                                       : make_float4(0.f, 0.f, 0.f, 0.f);
            const __half* hq = reinterpret_cast<const __half*>(&rq);
            const __half* hd = reinterpret_cast<const __half*>(&rd);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                Qs [(kb + i) * BR + r] = hq[i];
                dOs[(kb + i) * BR + r] = hd[i];
            }
        }
        __syncthreads();

        // ---- S = Q K^T and dP = dO V^T, both contracting over k ----
        float sv[C::TC][TR], dp[C::TC][TR];
#pragma unroll
        for (int c = 0; c < C::TC; ++c)
#pragma unroll
            for (int i = 0; i < TR; ++i) { sv[c][i] = 0.f; dp[c][i] = 0.f; }

        if constexpr (QDRAIN == 0) {
#pragma unroll 4
            for (int k = 0; k < D; ++k) {
                __half2 qh[TR / 2], oh[TR / 2];
                *reinterpret_cast<float4*>(qh) =
                    *reinterpret_cast<const float4*>(&Qs[k * BR + ty * TR]);
                *reinterpret_cast<float4*>(oh) =
                    *reinterpret_cast<const float4*>(&dOs[k * BR + ty * TR]);
                float qf[TR], of[TR];
#pragma unroll
                for (int i = 0; i < TR / 2; ++i) {
                    qf[2 * i] = __low2float(qh[i]);  qf[2 * i + 1] = __high2float(qh[i]);
                    of[2 * i] = __low2float(oh[i]);  of[2 * i + 1] = __high2float(oh[i]);
                }
#pragma unroll
                for (int c = 0; c < C::TC; ++c) {
                    const float kf = __half2float(Ks[k * BC + tx * C::TC + c]);
                    const float vf = __half2float(Vs[k * BC + tx * C::TC + c]);
#pragma unroll
                    for (int i = 0; i < TR; ++i) {
                        sv[c][i] = fmaf(qf[i], kf, sv[c][i]);
                        dp[c][i] = fmaf(of[i], vf, dp[c][i]);
                    }
                }
            }
        } else {
#pragma unroll 1
            for (int kb = 0; kb < D; kb += QDRAIN) {
                __half2 s2[C::TC][TR / 2], p2[C::TC][TR / 2];
#pragma unroll
                for (int c = 0; c < C::TC; ++c)
#pragma unroll
                    for (int i = 0; i < TR / 2; ++i) {
                        s2[c][i] = __float2half2_rn(0.f);
                        p2[c][i] = __float2half2_rn(0.f);
                    }
#pragma unroll
                for (int ko = 0; ko < QDRAIN; ++ko) {
                    const int k = kb + ko;
                    __half2 qh[TR / 2], oh[TR / 2];
                    *reinterpret_cast<float4*>(qh) =
                        *reinterpret_cast<const float4*>(&Qs[k * BR + ty * TR]);
                    *reinterpret_cast<float4*>(oh) =
                        *reinterpret_cast<const float4*>(&dOs[k * BR + ty * TR]);
#pragma unroll
                    for (int c = 0; c < C::TC; ++c) {
                        const __half2 k2 = __half2half2(Ks[k * BC + tx * C::TC + c]);
                        const __half2 v2 = __half2half2(Vs[k * BC + tx * C::TC + c]);
#pragma unroll
                        for (int i = 0; i < TR / 2; ++i) {
                            s2[c][i] = __hfma2(qh[i], k2, s2[c][i]);
                            p2[c][i] = __hfma2(oh[i], v2, p2[c][i]);
                        }
                    }
                }
#pragma unroll
                for (int c = 0; c < C::TC; ++c)
#pragma unroll
                    for (int i = 0; i < TR / 2; ++i) {
                        sv[c][2*i]   += __low2float (s2[c][i]);
                        sv[c][2*i+1] += __high2float(s2[c][i]);
                        dp[c][2*i]   += __low2float (p2[c][i]);
                        dp[c][2*i+1] += __high2float(p2[c][i]);
                    }
            }
        }

        // ---- P and dS ----
#pragma unroll
        for (int c = 0; c < C::TC; ++c) {
            const int col = j0 + tx * C::TC + c;
#pragma unroll
            for (int i = 0; i < TR; ++i) {
                const int row = i0 + ty * TR + i;
                // A padded row or column must contribute exactly zero to dV and
                // dK, or it corrupts the columns that are real.
                const bool dead = (BOUNDS && (row >= N || col >= N)) || (CAUSAL && col > row);
                const int  rs   = dead ? 0 : row;   // keep Lse/Dv reads in range
                const float p = dead ? 0.f : __expf(sv[c][i] * scale - Lseg[rs]);
                sv[c][i] = p;                                          // P
                dp[c][i] = dead ? 0.f : p * (dp[c][i] - Dvg[rs]);      // dS
            }
        }

        __syncthreads();   // last reads of the previous iteration's Ps/dSs
#pragma unroll
        for (int c = 0; c < C::TC; ++c) {
            __half pc[TR], sc[TR];
#pragma unroll
            for (int i = 0; i < TR; ++i) {
                pc[i] = __float2half(sv[c][i]);
                sc[i] = __float2half(dp[c][i]);
            }
            const int col = tx * C::TC + c;
            *reinterpret_cast<float4*>(&Ps [col * BR + ty * TR]) =
                *reinterpret_cast<const float4*>(pc);
            *reinterpret_cast<float4*>(&dSs[col * BR + ty * TR]) =
                *reinterpret_cast<const float4*>(sc);
        }
        __syncthreads();

        // ---- dV_j += P^T dO_i and dK_j += dS^T Q_i, contracting over r ----
        // r is contiguous in every one of these four tiles, so the contraction
        // reads 16 bytes at a time and both accumulators advance together.
#pragma unroll 1
        for (int r0 = 0; r0 < BR; r0 += 8) {
            __half2 ph[C::TCV][4], sh[C::TCV][4];
#pragma unroll
            for (int c = 0; c < C::TCV; ++c) {
                *reinterpret_cast<float4*>(ph[c]) =
                    *reinterpret_cast<const float4*>(&Ps [(cb + c) * BR + r0]);
                *reinterpret_cast<float4*>(sh[c]) =
                    *reinterpret_cast<const float4*>(&dSs[(cb + c) * BR + r0]);
            }
            // Convert ONCE per fragment, outside the pairing loop. Written the
            // naive way this loop issues two F2F for every FFMA -- the fp16
            // operands get re-expanded for each (c, t) pair -- and it is the
            // conversions, not the multiplies or the loads, that dominate.
            // Every fp32 operand here starts life as fp16, and on Pascal a
            // __half2float is a real instruction (HADD2.F32 x.H0_H0, -RZ) --
            // they were 13% of the whole kernel. H2_DVDK keeps the contraction
            // in __hfma2 and drains to fp32 every 8 rows instead: half the
            // multiplies and a third of the conversions. Safe here in a way it
            // is NOT in the outer products -- P is in [0,1] against
            // unit-variance dO, and the fp16 run is only 8 terms long.
            float pf[C::TCV][8], sfv[C::TCV][8];
            if constexpr (!H2_DVDK) {
#pragma unroll
                for (int c = 0; c < C::TCV; ++c)
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        pf [c][2*u]   = __low2float (ph[c][u]);
                        pf [c][2*u+1] = __high2float(ph[c][u]);
                        sfv[c][2*u]   = __low2float (sh[c][u]);
                        sfv[c][2*u+1] = __high2float(sh[c][u]);
                    }
            }
#pragma unroll
            for (int t = 0; t < C::TKV; ++t) {
                __half2 oh[4], qh[4];
                *reinterpret_cast<float4*>(oh) =
                    *reinterpret_cast<const float4*>(&dOs[(kb0 + t) * BR + r0]);
                *reinterpret_cast<float4*>(qh) =
                    *reinterpret_cast<const float4*>(&Qs [(kb0 + t) * BR + r0]);
                if constexpr (H2_DVDK) {
#pragma unroll
                    for (int c = 0; c < C::TCV; ++c) {
                        __half2 av2 = __float2half2_rn(0.f), ak2 = __float2half2_rn(0.f);
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            av2 = __hfma2(ph[c][u], oh[u], av2);
                            ak2 = __hfma2(sh[c][u], qh[u], ak2);
                        }
                        dv_acc[c][t] += __low2float(av2) + __high2float(av2);
                        dk_acc[c][t] += __low2float(ak2) + __high2float(ak2);
                    }
                } else {
                    float of[8], qf[8];
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        of[2*u] = __low2float(oh[u]);  of[2*u+1] = __high2float(oh[u]);
                        qf[2*u] = __low2float(qh[u]);  qf[2*u+1] = __high2float(qh[u]);
                    }
#pragma unroll
                    for (int c = 0; c < C::TCV; ++c) {
                        float av = 0.f, ak = 0.f;
#pragma unroll
                        for (int u = 0; u < 8; ++u) {
                            av = fmaf(pf [c][u], of[u], av);
                            ak = fmaf(sfv[c][u], qf[u], ak);
                        }
                        dv_acc[c][t] += av;
                        dk_acc[c][t] += ak;
                    }
                }
            }
        }

        // ---- dQ_i += dS K_j * scale, contracting over c ----
        float dq[C::TKQ][TR];
#pragma unroll
        for (int t = 0; t < C::TKQ; ++t)
#pragma unroll
            for (int i = 0; i < TR; ++i) dq[t][i] = 0.f;

        if constexpr (QDRAIN == 0) {
#pragma unroll 4
            for (int c = 0; c < BC; ++c) {
                __half2 sh[TR / 2];
                *reinterpret_cast<float4*>(sh) =
                    *reinterpret_cast<const float4*>(&dSs[c * BR + ty * TR]);
                float sfv[TR];
#pragma unroll
                for (int i = 0; i < TR / 2; ++i) {
                    sfv[2 * i] = __low2float(sh[i]); sfv[2 * i + 1] = __high2float(sh[i]);
                }
#pragma unroll
                for (int t = 0; t < C::TKQ; ++t) {
                    const float kf = __half2float(Ks[(tx * C::TKQ + t) * BC + c]);
#pragma unroll
                    for (int i = 0; i < TR; ++i) dq[t][i] = fmaf(sfv[i], kf, dq[t][i]);
                }
            }
        } else {
#pragma unroll 1
            for (int cb = 0; cb < BC; cb += QDRAIN) {
                __half2 dq2[C::TKQ][TR / 2];
#pragma unroll
                for (int t = 0; t < C::TKQ; ++t)
#pragma unroll
                    for (int i = 0; i < TR / 2; ++i) dq2[t][i] = __float2half2_rn(0.f);
#pragma unroll
                for (int co = 0; co < QDRAIN; ++co) {
                    const int c = cb + co;
                    __half2 sh[TR / 2];
                    *reinterpret_cast<float4*>(sh) =
                        *reinterpret_cast<const float4*>(&dSs[c * BR + ty * TR]);
#pragma unroll
                    for (int t = 0; t < C::TKQ; ++t) {
                        const __half2 k2 = __half2half2(Ks[(tx * C::TKQ + t) * BC + c]);
#pragma unroll
                        for (int i = 0; i < TR / 2; ++i)
                            dq2[t][i] = __hfma2(sh[i], k2, dq2[t][i]);
                    }
                }
#pragma unroll
                for (int t = 0; t < C::TKQ; ++t)
#pragma unroll
                    for (int i = 0; i < TR / 2; ++i) {
                        dq[t][2*i]   += __low2float (dq2[t][i]);
                        dq[t][2*i+1] += __high2float(dq2[t][i]);
                    }
            }
        }
#pragma unroll
        for (int i = 0; i < TR; ++i) {
            const int gr = i0 + ty * TR + i;
            if (BOUNDS && gr >= N) continue;
#pragma unroll
            for (int t = 0; t < C::TKQ; ++t) {
                float* slot = &dQg[(size_t)gr * D + tx * C::TKQ + t];
                if constexpr (NO_ATOMIC) *slot = dq[t][i] * scale;
                else                     atomicAdd(slot, dq[t][i] * scale);
            }
        }
      }
    }

    // ---- dK and dV leave once, straight from registers ----
#pragma unroll
    for (int c = 0; c < C::TCV; ++c) {
        if (BOUNDS && j0 + cb + c >= N) continue;
#pragma unroll
        for (int t = 0; t < C::TKV; ++t) {
            const size_t idx = (size_t)(j0 + cb + c) * D + kb0 + t;
            if constexpr (std::is_same<GOutT, __half>::value) {
                dV[idx] = __float2half(dv_acc[c][t]);
                dK[idx] = __float2half(dk_acc[c][t] * scale);
            } else {
                dV[idx] = dv_acc[c][t];
                dK[idx] = dk_acc[c][t] * scale;
            }
        }
    }
}

template <int BR, int BC, int D, int THREADS, bool CAUSAL = false,
          bool NO_ATOMIC = false, int QDRAIN = 0, typename GOutT = float,
          bool H2_DVDK = false>
void launch(const __half* Q, const __half* K, const __half* V, const __half* dO,
            const __half* O, float* Lse, float* Dv,
            float* dQ, GOutT* dK, GOutT* dV, int BH, int N, float scale,
            int Hq = 0, int GROUP = 1) {
    if (Hq == 0) Hq = BH;
    const int BHkv = BH / GROUP;
    bwd_preprocess<D><<<dim3(N, BH), 32>>>(O, dO, Dv, N);
    CUDA_CHECK(cudaMemsetAsync(dQ, 0, (size_t)BH * N * D * sizeof(float)));
    const dim3 grid((N + BC - 1) / BC, BHkv);
    if (N % BR == 0 && N % BC == 0)
        bwd_kernel<BR, BC, D, THREADS, CAUSAL, NO_ATOMIC, QDRAIN, GOutT, false, H2_DVDK>
            <<<grid, THREADS>>>(Q, K, V, dO, Lse, Dv, dQ, dK, dV, N, scale, Hq, GROUP);
    else
        bwd_kernel<BR, BC, D, THREADS, CAUSAL, NO_ATOMIC, QDRAIN, GOutT, true, H2_DVDK>
            <<<grid, THREADS>>>(Q, K, V, dO, Lse, Dv, dQ, dK, dV, N, scale, Hq, GROUP);
}

}  // namespace flash_bwd
