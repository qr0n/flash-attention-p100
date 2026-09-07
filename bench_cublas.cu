// Phase 1.1 — cuBLAS baselines.
//
// Everything downstream is scored against these numbers, so they are measured
// on the exact shapes the attention kernel will face, not just on a square.
//
//   square  M=N=K       : the realistic compute ceiling
//   QK^T    MxNxK=S,S,d : skinny-K, one matmul of the attention inner loop
//   PV      MxNxK=S,d,S : skinny-N, the other one
//
// fp16- and fp32-accumulate variants are both measured. The gap between them
// is what the Phase 2 precision strategy is spending or saving.

#include "common.cuh"

namespace {

struct Shape {
    const char* tag;
    int M, N, K;
};

enum class Acc { F16, F32 };

// Row-major C(MxN) = A(MxK) * B(KxN) via the standard column-major swap:
// cuBLAS computes C^T = B^T * A^T with no transposes requested.
double run_gemm(cublasHandle_t h, Acc acc, int M, int N, int K,
                const __half* dA, const __half* dB, __half* dC) {
    auto body = [&] {
        if (acc == Acc::F16) {
            const __half alpha = __float2half(1.f), beta = __float2half(0.f);
            CUBLAS_CHECK(cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                     &alpha, dB, N, dA, K, &beta, dC, N));
        } else {
            const float alpha = 1.f, beta = 0.f;
            CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                      &alpha, dB, CUDA_R_16F, N,
                                      dA, CUDA_R_16F, K, &beta,
                                      dC, CUDA_R_16F, N, CUDA_R_32F,
                                      CUBLAS_GEMM_DEFAULT));
        }
    };
    // Big shapes are slow enough that 20 iterations is wasteful; small ones
    // need the repetitions to escape timer granularity.
    double work = 2.0 * M * N * K;
    int iters = work > 5e10 ? 10 : 50;
    return time_ms(body, 5, iters);
}

void validate_layout(cublasHandle_t h) {
    const int M = 512, N = 384, K = 256;   // deliberately non-square
    std::vector<__half> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
    fill_normal(A, 1); fill_normal(B, 2);

    __half *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, A.size() * 2));
    CUDA_CHECK(cudaMalloc(&dB, B.size() * 2));
    CUDA_CHECK(cudaMalloc(&dC, C.size() * 2));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * 2, cudaMemcpyHostToDevice));

    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                              &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K,
                              &beta, dC, CUDA_R_16F, N, CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * 2, cudaMemcpyDeviceToHost));

    std::vector<double> ref;
    gemm_ref_f64(A, B, ref, M, N, K);
    ErrStats e = compare(C, ref, std::sqrt((double)K));

    std::printf("layout check  (%d,%d,%d) fp32-acc : max_rel %.3e  rms_rel %.3e  nan %d  -> %s\n\n",
                M, N, K, e.max_rel, e.rms_rel, e.n_nan,
                (e.max_rel < 5e-3 && e.n_nan == 0) ? "OK" : "WRONG");

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
}

}  // namespace

int main() {
    print_device_banner();
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));

    validate_layout(h);

    const Shape shapes[] = {
        {"square-2048",  2048, 2048, 2048},
        {"square-4096",  4096, 4096, 4096},
        {"square-8192",  8192, 8192, 8192},
        {"QK^T  S=2048 d=64",  2048, 2048,   64},
        {"QK^T  S=4096 d=64",  4096, 4096,   64},
        {"QK^T  S=4096 d=128", 4096, 4096,  128},
        {"PV    S=2048 d=64",  2048,   64, 2048},
        {"PV    S=4096 d=64",  4096,   64, 4096},
        {"PV    S=4096 d=128", 4096,  128, 4096},
    };

    std::printf("%-20s %10s %10s %10s %10s %8s\n",
                "shape", "ms(f16)", "TF(f16)", "ms(f32)", "TF(f32)", "f16/f32");
    std::printf("%s\n", std::string(74, '-').c_str());

    for (const Shape& s : shapes) {
        std::vector<__half> A((size_t)s.M * s.K), B((size_t)s.K * s.N);
        fill_normal(A, 11); fill_normal(B, 22);

        __half *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, A.size() * 2));
        CUDA_CHECK(cudaMalloc(&dB, B.size() * 2));
        CUDA_CHECK(cudaMalloc(&dC, (size_t)s.M * s.N * 2));
        CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * 2, cudaMemcpyHostToDevice));

        double ms16 = run_gemm(h, Acc::F16, s.M, s.N, s.K, dA, dB, dC);
        double ms32 = run_gemm(h, Acc::F32, s.M, s.N, s.K, dA, dB, dC);

        std::printf("%-20s %10.3f %10.2f %10.3f %10.2f %8.2fx\n", s.tag,
                    ms16, gemm_tflops(s.M, s.N, s.K, ms16),
                    ms32, gemm_tflops(s.M, s.N, s.K, ms32), ms32 / ms16);

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    }

    CUBLAS_CHECK(cublasDestroy(h));
    return 0;
}
