// Phase 2/3 — FlashAttention-2 forward pass for GP100 (sm_60).
// Templated on head dim, block shape, thread count, causality and the two
// precision knobs. No tensor cores exist here, so both matmuls are
// register-blocked __hfma2 outer products, the same shape of code as Phase 1.
//
// Thread mapping. THREADS threads are cut into ROWG row groups of TXR threads.
// A thread owns TR = 8 consecutive rows of the Q block for its whole lifetime,
// which is what keeps the softmax statistics (m, l) thread-local except for one
// shuffle reduction across the TXR threads sharing those rows.
//
//   S tile per thread : 8 rows x TC  columns of the BR x BC score block
//   O tile per thread : 8 rows x TCO columns of the BR x D  output block
//
// half2 packs along ROWS, not columns: rows come in pairs (2i, 2i+1) and are
// contiguous in the transposed shared layouts, so every fragment read is a
// 16-byte load and TC is free to be odd.
//
// Precision knobs, both measured in BENCH.log:
//   DRAIN   how many k-steps accumulate in fp16 before folding into an fp32
//           sum. DRAIN == D is pure fp16. Fixes the SUM rounding only.
//   FP32MAC forms each product with fmaf on converted halves instead of
//           __hfma2. Fixes the PRODUCT rounding, which draining cannot.

#pragma once
#include "common.cuh"

namespace flash {

constexpr int TR = 8;   // rows per thread, fixed

template <int BR, int BC, int D, int THREADS>
struct Cfg {
    static constexpr int ROWG = BR / TR;
    static constexpr int TXR  = THREADS / ROWG;
    static constexpr int TC   = BC / TXR;
    static constexpr int TCO  = D / TXR;
    static constexpr int QVEC = BR * D / 8 / THREADS;   // float4s per thread
    static constexpr int KVEC = BC * D / 8 / THREADS;

    // Shared: Q^T [D][BR]; K^T [D][BC] aliased with P^T [BC][BR]; V [BC][D].
    static constexpr int SM_Q  = D * BR;
    static constexpr int SM_KP = (D * BC > BC * BR) ? D * BC : BC * BR;
    static constexpr int SM_V  = BC * D;
    static constexpr int SMEM  = (SM_Q + SM_KP + SM_V) * 2;

    static_assert(BR % TR == 0, "BR must be a multiple of 8");
    static_assert(THREADS % ROWG == 0, "row groups must divide the block");
    static_assert(BC % TXR == 0 && TC >= 1, "BC too small for this BR/THREADS");
    static_assert(D % TXR == 0, "TXR must divide the head dim");
    static_assert(TXR <= 32, "row-group reduction must stay inside one warp");
    static_assert(QVEC >= 1 && KVEC >= 1, "tile too small for this thread count");
    static_assert(SMEM <= 49152, "over the 48 KB per-block shared memory cap");
};

// Reduce across the TXR contiguous lanes sharing a row group. XOR shuffles with
// offsets below TXR never leave the group, so no mask juggling is needed.
template <int TXR, class Op>
__device__ __forceinline__ float row_reduce(float v, Op op) {
#pragma unroll
    for (int off = TXR / 2; off > 0; off >>= 1)
        v = op(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// BOUNDS enables ragged-tail handling for an N that is not a multiple of the
// block sizes. It costs a few percent, so `launch` picks the aligned
// instantiation when it can: correctness for any N, full speed for aligned.
template <int BR, int BC, int D, int THREADS, bool CAUSAL, int DRAIN,
          bool FP32MAC, bool BOUNDS>
__global__ __launch_bounds__(THREADS) void fwd_kernel(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O,
    float* __restrict__ Lse, int N, float scale, int Hq, int GROUP) {

    using C = Cfg<BR, BC, D, THREADS>;
    __shared__ __align__(16) __half Qs[C::SM_Q];    // [k][r]
    __shared__ __align__(16) __half KP[C::SM_KP];   // [k][c]  then  [c][r]
    __shared__ __align__(16) __half Vs[C::SM_V];    // [c][k]

    const int tid = threadIdx.x;
    const int ty  = tid / C::TXR;
    const int tx  = tid % C::TXR;
    const int q0  = blockIdx.x * BR;

    // Grouped-query attention: blockIdx.y indexes a Q head, and several Q heads
    // share one K/V head. GROUP == 1 collapses this to ordinary MHA.
    const int bh    = blockIdx.y;                       // over B*Hq
    const int kv_bh = (bh / Hq) * (Hq / GROUP) + (bh % Hq) / GROUP;
    Q += (size_t)bh * N * D;
    O += (size_t)bh * N * D;
    K += (size_t)kv_bh * N * D;
    V += (size_t)kv_bh * N * D;

    // ---- load the Q block once, transposed. NOT pre-scaled ----
    // Pre-scaling Q by 1/sqrt(d) in fp16 rounds every Q entry unless the scale
    // is a power of two. 1/sqrt(64) = 0.125 is; 1/sqrt(128) is not, and that
    // single rounding was worth ~1e-1 of relative error at d=128 with large
    // inputs -- it survived even fp32 MACs, because it happens before them.
    // Applying the scale to the fp32 logit afterwards is exact and free.
    // The unscaled dot product stays far inside fp16 range: ~800 for the
    // worst regime measured here against a 65504 limit.
    {
#pragma unroll
        for (int v = 0; v < C::QVEC; ++v) {
            const int flat = v * THREADS + tid;
            const int r = flat / (D / 8), kb = (flat % (D / 8)) * 8;
            const int gr = q0 + r;
            const float4 raw = (!BOUNDS || gr < N)
                ? *reinterpret_cast<const float4*>(&Q[(size_t)gr * D + kb])
                : make_float4(0.f, 0.f, 0.f, 0.f);
            const __half* h = reinterpret_cast<const __half*>(&raw);
#pragma unroll
            for (int i = 0; i < 8; ++i) Qs[(kb + i) * BR + r] = h[i];
        }
    }

    float m[TR], l[TR], acc[C::TCO][TR];
#pragma unroll
    for (int i = 0; i < TR; ++i) { m[i] = -INFINITY; l[i] = 0.f; }
#pragma unroll
    for (int c = 0; c < C::TCO; ++c)
#pragma unroll
        for (int i = 0; i < TR; ++i) acc[c][i] = 0.f;

    __syncthreads();

    // Causal: a Q block never needs a K block that starts past its last row.
    // BR is a multiple of BC, so this bound stays on a block boundary and the
    // whole upper triangle is skipped without ever being loaded.
    const int j_end = CAUSAL ? min(N, q0 + BR) : N;   // loop is ceil'd by the += BC step

    for (int j0 = 0; j0 < j_end; j0 += BC) {
        __syncthreads();   // previous iteration's reads of KP / Vs are done

        // ---- stage K (transposed) and V (straight) ----
#pragma unroll
        for (int v = 0; v < C::KVEC; ++v) {
            const int flat = v * THREADS + tid;
            const int c = flat / (D / 8), kb = (flat % (D / 8)) * 8;
            const int gc = j0 + c;
            const float4 raw = (!BOUNDS || gc < N)
                ? *reinterpret_cast<const float4*>(&K[(size_t)gc * D + kb])
                : make_float4(0.f, 0.f, 0.f, 0.f);
            const __half* h = reinterpret_cast<const __half*>(&raw);
#pragma unroll
            for (int i = 0; i < 8; ++i) KP[(kb + i) * BC + c] = h[i];
        }
#pragma unroll
        for (int v = 0; v < C::KVEC; ++v) {
            const int flat = v * THREADS + tid;
            const int c = flat / (D / 8), kb = (flat % (D / 8)) * 8;
            const int gc2 = j0 + c;
            *reinterpret_cast<float4*>(&Vs[c * D + kb]) = (!BOUNDS || gc2 < N)
                ? *reinterpret_cast<const float4*>(&V[(size_t)gc2 * D + kb])
                : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        __syncthreads();

        // ---- S = Q_block . K_block^T ----
        float sf[C::TC][TR];
#pragma unroll
        for (int c = 0; c < C::TC; ++c)
#pragma unroll
            for (int i = 0; i < TR; ++i) sf[c][i] = 0.f;

        if constexpr (FP32MAC) {
#pragma unroll 8
            for (int k = 0; k < D; ++k) {
                __half2 qh[TR / 2];
                *reinterpret_cast<float4*>(qh) =
                    *reinterpret_cast<const float4*>(&Qs[k * BR + ty * TR]);
                float qf[TR];
#pragma unroll
                for (int i = 0; i < TR / 2; ++i) {
                    qf[2 * i]     = __low2float(qh[i]);
                    qf[2 * i + 1] = __high2float(qh[i]);
                }
#pragma unroll
                for (int c = 0; c < C::TC; ++c) {
                    const float kf = __half2float(KP[k * BC + tx * C::TC + c]);
#pragma unroll
                    for (int i = 0; i < TR; ++i) sf[c][i] = fmaf(qf[i], kf, sf[c][i]);
                }
            }
        } else {
#pragma unroll 1
            for (int kb = 0; kb < D; kb += DRAIN) {
                __half2 s[C::TC][TR / 2];
#pragma unroll
                for (int c = 0; c < C::TC; ++c)
#pragma unroll
                    for (int i = 0; i < TR / 2; ++i) s[c][i] = __float2half2_rn(0.f);

#pragma unroll
                for (int ko = 0; ko < DRAIN; ++ko) {
                    const int k = kb + ko;
                    __half2 qf[TR / 2];
                    *reinterpret_cast<float4*>(qf) =
                        *reinterpret_cast<const float4*>(&Qs[k * BR + ty * TR]);
                    __half kf[C::TC];
#pragma unroll
                    for (int c = 0; c < C::TC; ++c) kf[c] = KP[k * BC + tx * C::TC + c];
#pragma unroll
                    for (int c = 0; c < C::TC; ++c) {
                        const __half2 k2 = __half2half2(kf[c]);
#pragma unroll
                        for (int i = 0; i < TR / 2; ++i) s[c][i] = __hfma2(qf[i], k2, s[c][i]);
                    }
                }
#pragma unroll
                for (int c = 0; c < C::TC; ++c)
#pragma unroll
                    for (int i = 0; i < TR / 2; ++i) {
                        sf[c][2 * i]     += __low2float(s[c][i]);
                        sf[c][2 * i + 1] += __high2float(s[c][i]);
                    }
            }
        }

        // ---- apply 1/sqrt(d) here, in fp32, exactly ----
#pragma unroll
        for (int c = 0; c < C::TC; ++c)
#pragma unroll
            for (int i = 0; i < TR; ++i) sf[c][i] *= scale;

        // ---- masking: causal, and the ragged tail when N is not a multiple
        // of BC. A padded column carries K=0, which gives S=0 -- and exp(0-m)
        // is NOT zero, so the tail must be masked explicitly, not left to the
        // zero fill.
        const bool tail = BOUNDS && (j0 + BC > N);
        if (CAUSAL || tail) {
            if (tail || j0 + BC > q0) {
#pragma unroll
                for (int c = 0; c < C::TC; ++c) {
                    const int col = j0 + tx * C::TC + c;
                    const bool oob = BOUNDS && col >= N;
#pragma unroll
                    for (int i = 0; i < TR; ++i)
                        if (oob || (CAUSAL && col > q0 + ty * TR + i))
                            sf[c][i] = -INFINITY;
                }
            }
        }

        // ---- online softmax, fp32 ----
        float corr[TR];
#pragma unroll
        for (int i = 0; i < TR; ++i) {
            float mx = -INFINITY;
#pragma unroll
            for (int c = 0; c < C::TC; ++c) mx = fmaxf(mx, sf[c][i]);
            mx = row_reduce<C::TXR>(mx, [](float a, float b) { return fmaxf(a, b); });

            const float m_new = fmaxf(m[i], mx);
            corr[i] = __expf(m[i] - m_new);      // 0 on the first block, not NaN

            float sum = 0.f;
#pragma unroll
            for (int c = 0; c < C::TC; ++c) {
                sf[c][i] = __expf(sf[c][i] - m_new);
                sum += sf[c][i];
            }
            sum = row_reduce<C::TXR>(sum, [](float a, float b) { return a + b; });

            l[i] = l[i] * corr[i] + sum;
            m[i] = m_new;
        }
#pragma unroll
        for (int c = 0; c < C::TCO; ++c)
#pragma unroll
            for (int i = 0; i < TR; ++i) acc[c][i] *= corr[i];

        __syncthreads();   // every read of K is retired; KP now becomes P

        // ---- publish P transposed. A thread's 8 rows are contiguous in the
        // [c][r] layout, so one column of P leaves as a single 16-byte store.
#pragma unroll
        for (int c = 0; c < C::TC; ++c) {
            __half col[TR];
#pragma unroll
            for (int i = 0; i < TR; ++i) col[i] = __float2half(sf[c][i]);
            *reinterpret_cast<float4*>(&KP[(tx * C::TC + c) * BR + ty * TR]) =
                *reinterpret_cast<const float4*>(col);
        }
        __syncthreads();

        // ---- O += P . V, fp16 accumulate over BC only, drained to fp32 ----
        // Verified safe: P is in [0,1] and sums to <=1 across the block, so
        // regime 3 (near-uniform attention) stresses this sum and still lands
        // at 4.5e-3. No fp32 tier is needed here.
        __half2 op[C::TCO][TR / 2];
#pragma unroll
        for (int c = 0; c < C::TCO; ++c)
#pragma unroll
            for (int i = 0; i < TR / 2; ++i) op[c][i] = __float2half2_rn(0.f);

#pragma unroll 8
        for (int jj = 0; jj < BC; ++jj) {
            __half2 pf[TR / 2];
            *reinterpret_cast<float4*>(pf) =
                *reinterpret_cast<const float4*>(&KP[jj * BR + ty * TR]);
            __half vf[C::TCO];
#pragma unroll
            for (int c = 0; c < C::TCO; ++c) vf[c] = Vs[jj * D + tx * C::TCO + c];
#pragma unroll
            for (int c = 0; c < C::TCO; ++c) {
                const __half2 v2 = __half2half2(vf[c]);
#pragma unroll
                for (int i = 0; i < TR / 2; ++i) op[c][i] = __hfma2(pf[i], v2, op[c][i]);
            }
        }
#pragma unroll
        for (int c = 0; c < C::TCO; ++c)
#pragma unroll
            for (int i = 0; i < TR / 2; ++i) {
                acc[c][2 * i]     += __low2float(op[c][i]);
                acc[c][2 * i + 1] += __high2float(op[c][i]);
            }
    }

    // ---- normalise and store ----
    // Lse = m + log(l) is everything the backward pass needs to rebuild P
    // without re-running the online softmax: P = exp(S*scale - Lse).
    if (Lse != nullptr && tx == 0) {
#pragma unroll
        for (int i = 0; i < TR; ++i) {
            const int gr = q0 + ty * TR + i;
            if (!BOUNDS || gr < N) Lse[(size_t)bh * N + gr] = m[i] + __logf(l[i]);
        }
    }
#pragma unroll
    for (int i = 0; i < TR; ++i) {
        const int gr = q0 + ty * TR + i;
        if (BOUNDS && gr >= N) continue;
        const float inv = 1.f / l[i];
#pragma unroll
        for (int c = 0; c < C::TCO; ++c)
            O[(size_t)gr * D + tx * C::TCO + c] = __float2half(acc[c][i] * inv);
    }
}

template <int BR, int BC, int D, int THREADS, bool CAUSAL = false,
          int DRAIN = D, bool FP32MAC = false>
void launch(const __half* Q, const __half* K, const __half* V, __half* O,
            int BH, int N, float scale, float* Lse = nullptr,
            int Hq = 0, int GROUP = 1) {
    if (Hq == 0) Hq = BH;                 // single batch unless told otherwise
    dim3 grid((N + BR - 1) / BR, BH);
    if (N % BR == 0 && N % BC == 0)
        fwd_kernel<BR, BC, D, THREADS, CAUSAL, DRAIN, FP32MAC, false>
            <<<grid, THREADS>>>(Q, K, V, O, Lse, N, scale, Hq, GROUP);
    else
        fwd_kernel<BR, BC, D, THREADS, CAUSAL, DRAIN, FP32MAC, true>
            <<<grid, THREADS>>>(Q, K, V, O, Lse, N, scale, Hq, GROUP);
}

}  // namespace flash
