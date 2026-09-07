// Phase 1.2 / 1.3 — half2 SIMT GEMM for GP100.
//
// Row-major C(MxN) = A(MxK) * B(KxN), fp16 in, fp16 accumulate.
// No tensor cores exist on sm_60, so every version here is a register-blocked
// outer product built out of __hfma2. Versions are kept side by side and all
// measured in one run, because the only claim worth making about an
// optimization is the delta it produced on this card.
//
// Accumulators are held as __half2 packed along N: acc[i][j] covers output
// columns (2j, 2j+1) of row i. One __hfma2 therefore retires two FMAs, which
// is the entire reason this runs at 2x the fp32 rate.

#include "common.cuh"

// ---------------------------------------------------------------------------
// Tiled half2 GEMM, templated on the K-step and on whether the next K-block is
// prefetched into registers while the current one is being multiplied.
//
// Pascal has no cp.async, so a register prefetch is the only way to overlap
// global latency with math: issue the loads for block k+1 before the MAC loop
// for block k, and commit them to shared memory afterwards.
//
// BM=128, BN=128, 8x8 per thread, 256 threads. Shared memory holds A
// transposed to [k][m] so that both fragment reads are 16-byte contiguous.
// ---------------------------------------------------------------------------
namespace tiled {

constexpr int BM = 128, BN = 128;
constexpr int TM = 8, TN = 8;
constexpr int THREADS = (BM / TM) * (BN / TN);   // 16x16 = 256

template <int BK>
struct Loader {
    // Each thread moves BK/2 halves of each tile, as BK/16 float4s.
    static constexpr int A_VEC = BM * BK / 8 / THREADS;
    static constexpr int B_VEC = BN * BK / 8 / THREADS;
    static_assert(A_VEC >= 1 && B_VEC >= 1, "tile too small for 256 threads");
};

template <int BK, bool PREFETCH>
__global__ __launch_bounds__(THREADS) void kernel(
    const __half* __restrict__ A, const __half* __restrict__ B,
    __half* __restrict__ C, int M, int N, int K) {

    using L = Loader<BK>;
    __shared__ __align__(16) __half As[BK][BM];   // transposed: [k][m]
    __shared__ __align__(16) __half Bs[BK][BN];   // [k][n]

    const int tid = threadIdx.x;
    const int ty  = tid / (BN / TN);
    const int tx  = tid % (BN / TN);
    const int bm  = blockIdx.y * BM;
    const int bn  = blockIdx.x * BN;

    // Fixed global->shared assignments. A is read along K (coalesced across
    // the 8 halves of one thread) and scattered transposed into shared.
    int a_row[L::A_VEC], a_col[L::A_VEC], b_row[L::B_VEC], b_col[L::B_VEC];
#pragma unroll
    for (int v = 0; v < L::A_VEC; ++v) {
        const int flat = v * THREADS + tid;
        a_row[v] = flat / (BK / 8);
        a_col[v] = (flat % (BK / 8)) * 8;
    }
#pragma unroll
    for (int v = 0; v < L::B_VEC; ++v) {
        const int flat = v * THREADS + tid;
        b_row[v] = flat / (BN / 8);
        b_col[v] = (flat % (BN / 8)) * 8;
    }

    __half2 acc[TM][TN / 2];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN / 2; ++j) acc[i][j] = __float2half2_rn(0.f);

    float4 ra[L::A_VEC], rb[L::B_VEC];

    auto global_load = [&](int k0) {
#pragma unroll
        for (int v = 0; v < L::A_VEC; ++v)
            ra[v] = *reinterpret_cast<const float4*>(
                &A[(size_t)(bm + a_row[v]) * K + k0 + a_col[v]]);
#pragma unroll
        for (int v = 0; v < L::B_VEC; ++v)
            rb[v] = *reinterpret_cast<const float4*>(
                &B[(size_t)(k0 + b_row[v]) * N + bn + b_col[v]]);
    };

    auto commit = [&]() {
#pragma unroll
        for (int v = 0; v < L::A_VEC; ++v) {
            const __half* h = reinterpret_cast<const __half*>(&ra[v]);
#pragma unroll
            for (int i = 0; i < 8; ++i) As[a_col[v] + i][a_row[v]] = h[i];
        }
#pragma unroll
        for (int v = 0; v < L::B_VEC; ++v)
            *reinterpret_cast<float4*>(&Bs[b_row[v]][b_col[v]]) = rb[v];
    };

    auto mac_block = [&]() {
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            __half  a_frag[TM];
            __half2 b_frag[TN / 2];
            *reinterpret_cast<float4*>(a_frag) =
                *reinterpret_cast<const float4*>(&As[kk][ty * TM]);
            *reinterpret_cast<float4*>(b_frag) =
                *reinterpret_cast<const float4*>(&Bs[kk][tx * TN]);
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const __half2 a2 = __half2half2(a_frag[i]);
#pragma unroll
                for (int j = 0; j < TN / 2; ++j)
                    acc[i][j] = __hfma2(a2, b_frag[j], acc[i][j]);
            }
        }
    };

    if (PREFETCH) {
        global_load(0);
        commit();
        __syncthreads();
        for (int k0 = BK; k0 < K; k0 += BK) {
            global_load(k0);        // issued before the math, retired after it
            mac_block();
            __syncthreads();
            commit();
            __syncthreads();
        }
        mac_block();
    } else {
        for (int k0 = 0; k0 < K; k0 += BK) {
            global_load(k0);
            commit();
            __syncthreads();
            mac_block();
            __syncthreads();
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
        *reinterpret_cast<float4*>(&C[(size_t)(bm + ty * TM + i) * N + bn + tx * TN]) =
            *reinterpret_cast<const float4*>(acc[i]);
}

template <int BK, bool PREFETCH>
void launch(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    dim3 grid(N / BN, M / BM), block(THREADS);
    kernel<BK, PREFETCH><<<grid, block>>>(A, B, C, M, N, K);
}

}  // namespace tiled

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
namespace {

struct Version {
    const char* name;
    void (*launch)(const __half*, const __half*, __half*, int, int, int);
};

const Version kVersions[] = {
    {"v1  128x128x16  single-buffered", tiled::launch<16, false>},
    {"v2  128x128x16  reg prefetch   ", tiled::launch<16, true>},
    {"v3  128x128x32  single-buffered", tiled::launch<32, false>},
    {"v4  128x128x32  reg prefetch   ", tiled::launch<32, true>},
};

bool validate(const Version& v) {
    const int M = 512, N = 384, K = 256;
    std::vector<__half> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
    fill_normal(A, 11);
    fill_normal(B, 22);

    __half *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, A.size() * 2));
    CUDA_CHECK(cudaMalloc(&dB, B.size() * 2));
    CUDA_CHECK(cudaMalloc(&dC, C.size() * 2));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, C.size() * 2));

    v.launch(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * 2, cudaMemcpyDeviceToHost));

    std::vector<double> ref;
    gemm_ref_f64(A, B, ref, M, N, K);
    // fp16 accumulation over K=256 terms of unit variance: the sum has scale
    // ~sqrt(K), so that is the right denominator floor for a relative error.
    ErrStats e = compare(C, ref, std::sqrt((double)K));

    const bool ok = e.max_rel < 5e-2 && e.n_nan == 0;
    std::printf("  validate (%d,%d,%d): max_abs %.4f  max_rel %.3e  rms_rel %.3e  nan %d  -> %s\n",
                M, N, K, e.max_abs, e.max_rel, e.rms_rel, e.n_nan, ok ? "OK" : "FAIL");
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const int S = (argc > 1) ? std::atoi(argv[1]) : 4096;
    print_device_banner();

    // cuBLAS reference at the benchmark shape, for the %cuBLAS column.
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));

    std::vector<__half> A((size_t)S * S), B((size_t)S * S);
    fill_normal(A, 11);
    fill_normal(B, 22);
    __half *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, A.size() * 2));
    CUDA_CHECK(cudaMalloc(&dB, B.size() * 2));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)S * S * 2));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * 2, cudaMemcpyHostToDevice));

    const __half alpha = __float2half(1.f), beta = __float2half(0.f);
    double ms_cublas = time_ms([&] {
        CUBLAS_CHECK(cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S,
                                 &alpha, dB, S, dA, S, &beta, dC, S));
    }, 5, 10);
    const double tf_cublas = gemm_tflops(S, S, S, ms_cublas);
    std::printf("\ncuBLAS Hgemm %d^3: %.3f ms  %.2f TFLOPS  (gate = %.2f)\n\n",
                S, ms_cublas, tf_cublas, 0.5 * tf_cublas);

    for (const Version& v : kVersions) {
        std::printf("%s\n", v.name);
        if (!validate(v)) { std::printf("  skipping benchmark, kernel is wrong\n\n"); continue; }
        double ms = time_ms([&] { v.launch(dA, dB, dC, S, S, S); }, 5, 10);
        double tf = gemm_tflops(S, S, S, ms);
        std::printf("  %d^3: %.3f ms  %.2f TFLOPS  %.1f%% of cuBLAS  %s\n\n",
                    S, ms, tf, 100.0 * tf / tf_cublas,
                    tf >= 0.5 * tf_cublas ? "[GATE PASS]" : "");
    }

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    CUBLAS_CHECK(cublasDestroy(h));
    return 0;
}
