// Shared harness for the FlashAttention-on-P100 project.
// Timing, fp64 CPU references and error metrics live here so that every
// kernel in every phase is judged by exactly the same yardstick.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <string>
#include <functional>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t s_ = (call);                                               \
        if (s_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(s_));                              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t s_ = (call);                                            \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__,         \
                         __LINE__, (int)s_);                                   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Timing. Warm up, then average over `iters` back-to-back launches inside one
// event pair, which keeps launch overhead out of the per-iteration number.
// ---------------------------------------------------------------------------
inline double time_ms(const std::function<void()>& body, int warmup = 5,
                      int iters = 20) {
    for (int i = 0; i < warmup; ++i) body();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < iters; ++i) body();
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    return (double)ms / iters;
}

inline double gemm_tflops(int M, int N, int K, double ms) {
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12;
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
inline void fill_normal(std::vector<__half>& v, unsigned seed, float scale = 1.f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (auto& x : v) x = __float2half(dist(rng) * scale);
}

// ---------------------------------------------------------------------------
// Row-major fp64 reference: C(MxN) = A(MxK) * B(KxN).
// Deliberately naive; only ever call it on small shapes.
// ---------------------------------------------------------------------------
inline void gemm_ref_f64(const std::vector<__half>& A, const std::vector<__half>& B,
                         std::vector<double>& C, int M, int N, int K) {
    C.assign((size_t)M * N, 0.0);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            double a = (double)__half2float(A[(size_t)i * K + k]);
            if (a == 0.0) continue;
            for (int j = 0; j < N; ++j)
                C[(size_t)i * N + j] += a * (double)__half2float(B[(size_t)k * N + j]);
        }
}

struct ErrStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double rms_rel = 0.0;
    int    n_nan   = 0;
};

// Relative error is measured against the magnitude of the reference value, with
// a floor so that entries near zero cannot manufacture an enormous ratio.
inline ErrStats compare(const std::vector<__half>& got, const std::vector<double>& ref,
                        double denom_floor = 1.0) {
    ErrStats e;
    double acc = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double g = (double)__half2float(got[i]);
        if (std::isnan(g) || std::isinf(g)) { ++e.n_nan; continue; }
        double d = std::fabs(g - ref[i]);
        double r = d / std::max(std::fabs(ref[i]), denom_floor);
        if (d > e.max_abs) e.max_abs = d;
        if (r > e.max_rel) e.max_rel = r;
        acc += r * r;
    }
    e.rms_rel = std::sqrt(acc / (double)ref.size());
    return e;
}

inline ErrStats compare(const std::vector<float>& got, const std::vector<double>& ref,
                        double denom_floor = 1.0) {
    ErrStats e;
    double acc = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double g = (double)got[i];
        if (std::isnan(g) || std::isinf(g)) { ++e.n_nan; continue; }
        const double d = std::fabs(g - ref[i]);
        const double r = d / std::max(std::fabs(ref[i]), denom_floor);
        if (d > e.max_abs) e.max_abs = d;
        if (r > e.max_rel) e.max_rel = r;
        acc += r * r;
    }
    e.rms_rel = std::sqrt(acc / (double)ref.size());
    return e;
}

inline void print_device_banner() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    std::printf("%s  (sm_%d%d, %d SMs, %.1f GB, %d MHz, smem/block %zu KB)\n",
                p.name, p.major, p.minor, p.multiProcessorCount,
                p.totalGlobalMem / 1e9, p.clockRate / 1000,
                p.sharedMemPerBlock / 1024);
}
