// Phase 2/3 driver — correctness across four input regimes at d=64 and d=128,
// causal and non-causal, then speed against an unfused fp16 attention built
// from cuBLAS plus a softmax kernel.
//
// The unfused path is the honest comparison: it is what you would otherwise
// run, and it is the thing whose N^2 score matrix the fused kernel exists to
// never materialise. Note it cannot skip the causal upper triangle — cuBLAS
// computes the whole product either way — which is where most of the causal
// win comes from.

#include "common.cuh"
#include "flash_fwd.cuh"

namespace {

// ---------------------------------------------------------------------------
// Unfused reference: cuBLAS QK^T -> softmax (+mask) -> PV.
// ---------------------------------------------------------------------------
__global__ void softmax_rows(__half* __restrict__ S, int N, float scale, bool causal) {
    extern __shared__ float red[];
    const int row = blockIdx.x;
    __half* r = S + ((size_t)blockIdx.y * N + row) * N;
    const int lim = causal ? row + 1 : N;

    float mx = -INFINITY;
    for (int j = threadIdx.x; j < lim; j += blockDim.x)
        mx = fmaxf(mx, __half2float(r[j]) * scale);
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();

    float sum = 0.f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        const float e = (j < lim) ? __expf(__half2float(r[j]) * scale - mx) : 0.f;
        r[j] = __float2half(e);
        sum += e;
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.f / red[0];
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        r[j] = __float2half(__half2float(r[j]) * inv);
}

void attention_unfused(cublasHandle_t h, const __half* Q, const __half* K,
                       const __half* V, __half* O, __half* S,
                       int BH, int N, int d, float scale, bool causal) {
    const __half one = __float2half(1.f), zero = __float2half(0.f);
    CUBLAS_CHECK(cublasHgemmStridedBatched(
        h, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d,
        &one, K, d, (long long)N * d, Q, d, (long long)N * d,
        &zero, S, N, (long long)N * N, BH));
    softmax_rows<<<dim3(N, BH), 256, 256 * sizeof(float)>>>(S, N, scale, causal);
    CUBLAS_CHECK(cublasHgemmStridedBatched(
        h, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N,
        &one, V, d, (long long)N * d, S, N, (long long)N * N,
        &zero, O, d, (long long)N * d, BH));
}

// ---------------------------------------------------------------------------
// fp64 CPU reference. Only ever called on small shapes.
// ---------------------------------------------------------------------------
void attention_ref_f64(const std::vector<__half>& Q, const std::vector<__half>& K,
                       const std::vector<__half>& V, std::vector<double>& O,
                       int BH, int N, int d, double scale, bool causal) {
    O.assign((size_t)BH * N * d, 0.0);
    std::vector<double> s(N);
    for (int b = 0; b < BH; ++b) {
        const size_t base = (size_t)b * N * d;
        for (int i = 0; i < N; ++i) {
            const int lim = causal ? i + 1 : N;
            double mx = -INFINITY;
            for (int j = 0; j < lim; ++j) {
                double acc = 0.0;
                for (int k = 0; k < d; ++k)
                    acc += (double)__half2float(Q[base + (size_t)i * d + k]) *
                           (double)__half2float(K[base + (size_t)j * d + k]);
                s[j] = acc * scale;
                if (s[j] > mx) mx = s[j];
            }
            double sum = 0.0;
            for (int j = 0; j < lim; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
            const double inv = 1.0 / sum;
            for (int j = 0; j < lim; ++j) {
                const double p = s[j] * inv;
                if (p == 0.0) continue;
                for (int k = 0; k < d; ++k)
                    O[base + (size_t)i * d + k] +=
                        p * (double)__half2float(V[base + (size_t)j * d + k]);
            }
        }
    }
}

// Output entries shrink as attention flattens, so a fixed denominator would
// report nonsense. Scale the relative error by the RMS of the reference.
ErrStats compare_vs_rms(const std::vector<__half>& got, const std::vector<double>& ref) {
    double acc = 0.0;
    for (double r : ref) acc += r * r;
    return compare(got, ref, std::sqrt(acc / (double)ref.size()));
}

struct Regime { const char* name; float scale; };
const Regime kRegimes[] = {
    {"1 N(0,1)      ", 1.00f},
    {"2 large x10   ", 10.0f},
    {"3 near-uniform", 0.05f},
    {"4 near-one-hot", 3.00f},
};

using LaunchFn = void (*)(const __half*, const __half*, const __half*, __half*, int, int, float, float*, int, int);
struct Variant { const char* name; LaunchFn fn; int d; bool causal; int smem; };

std::string regime_line(const Variant& v, int BH, int N) {
    const float scale = 1.f / std::sqrt((float)v.d);
    char buf[256]; std::string out;
    for (const Regime& rg : kRegimes) {
        std::vector<__half> Q((size_t)BH * N * v.d), K(Q.size()), V(Q.size()), O(Q.size());
        fill_normal(Q, 101, rg.scale);
        fill_normal(K, 202, rg.scale);
        fill_normal(V, 303, 1.0f);
        __half *dQ, *dK, *dV, *dO;
        CUDA_CHECK(cudaMalloc(&dQ, Q.size() * 2)); CUDA_CHECK(cudaMalloc(&dK, K.size() * 2));
        CUDA_CHECK(cudaMalloc(&dV, V.size() * 2)); CUDA_CHECK(cudaMalloc(&dO, O.size() * 2));
        CUDA_CHECK(cudaMemcpy(dQ, Q.data(), Q.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK, K.data(), K.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV, V.data(), V.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dO, 0, O.size() * 2));
        v.fn(dQ, dK, dV, dO, BH, N, scale, nullptr, BH, 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(O.data(), dO, O.size() * 2, cudaMemcpyDeviceToHost));
        std::vector<double> ref;
        attention_ref_f64(Q, K, V, ref, BH, N, v.d, scale, v.causal);
        ErrStats e = compare_vs_rms(O, ref);
        std::snprintf(buf, sizeof buf, " %8.1e%s", e.max_rel,
                      (e.max_rel < 5e-2 && e.n_nan == 0) ? " " : "*");
        out += buf;
        CUDA_CHECK(cudaFree(dQ)); CUDA_CHECK(cudaFree(dK));
        CUDA_CHECK(cudaFree(dV)); CUDA_CHECK(cudaFree(dO));
    }
    return out;
}

}  // namespace

#define V64(BR, BC, T, CAUS, DR, F32) \
    flash::launch<BR, BC, 64, T, CAUS, DR, F32>
#define V128(BR, BC, T, CAUS, DR, F32) \
    flash::launch<BR, BC, 128, T, CAUS, DR, F32>

int main() {
    print_device_banner();

    const Variant variants[] = {
      // d=64, non-causal
      {"d64  64x64  T128 drain16", V64(64,64,128,false,16,false),  64,false, flash::Cfg<64,64,64,128>::SMEM},
      {"d64  64x64  T256 drain16", V64(64,64,256,false,16,false),  64,false, flash::Cfg<64,64,64,256>::SMEM},
      {"d64  128x32 T128 drain16", V64(128,32,128,false,16,false), 64,false, flash::Cfg<128,32,64,128>::SMEM},
      {"d64  64x64  T128 fp32mac", V64(64,64,128,false,64,true),   64,false, flash::Cfg<64,64,64,128>::SMEM},
      // d=64, causal
      {"d64c 64x64  T128 drain16", V64(64,64,128,true,16,false),   64,true,  flash::Cfg<64,64,64,128>::SMEM},
      {"d64c 128x32 T128 drain16", V64(128,32,128,true,16,false),  64,true,  flash::Cfg<128,32,64,128>::SMEM},
      {"d64c 64x64  T128 fp32mac", V64(64,64,128,true,64,true),    64,true,  flash::Cfg<64,64,64,128>::SMEM},
      // d=128, non-causal
      {"d128 64x32  T256 drain16", V128(64,32,256,false,16,false), 128,false,flash::Cfg<64,32,128,256>::SMEM},
      {"d128 64x32  T128 drain16", V128(64,32,128,false,16,false), 128,false,flash::Cfg<64,32,128,128>::SMEM},
      {"d128 32x32  T128 drain16", V128(32,32,128,false,16,false), 128,false,flash::Cfg<32,32,128,128>::SMEM},
      {"d128 64x64  T256 drain16", V128(64,64,256,false,16,false), 128,false,flash::Cfg<64,64,128,256>::SMEM},
      {"d128 64x16  T128 drain16", V128(64,16,128,false,16,false), 128,false,flash::Cfg<64,16,128,128>::SMEM},
      {"d128 64x32  T128 drain64", V128(64,32,128,false,128,false),128,false,flash::Cfg<64,32,128,128>::SMEM},
      {"d128 64x32  T256 fp32mac", V128(64,32,256,false,128,true), 128,false,flash::Cfg<64,32,128,256>::SMEM},
      // d=128, causal
      {"d128c 64x32 T256 drain16", V128(64,32,256,true,16,false),  128,true, flash::Cfg<64,32,128,256>::SMEM},
      {"d128c 64x32 T256 fp32mac", V128(64,32,256,true,128,true),  128,true, flash::Cfg<64,32,128,256>::SMEM},
    };

    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));

    // ---------------- correctness ----------------
    std::printf("\n=== max_rel vs fp64 CPU reference (BH=2, N=256).  * = over 5e-2 ===\n");
    std::printf("%-26s %9s %9s %9s %9s   smem\n", "variant",
                "r1 N(0,1)", "r2 x10", "r3 unif", "r4 1hot");
    for (const Variant& v : variants)
        std::printf("%-26s%s   %d B\n", v.name, regime_line(v, 2, 256).c_str(), v.smem);

    // ---------------- speed ----------------
    const int BH = 16;
    for (int N : {2048, 4096}) {
        for (int d : {64, 128}) {
            std::vector<__half> H((size_t)BH * N * d);
            __half *dQ, *dK, *dV, *dO, *dS;
            CUDA_CHECK(cudaMalloc(&dQ, H.size() * 2)); CUDA_CHECK(cudaMalloc(&dK, H.size() * 2));
            CUDA_CHECK(cudaMalloc(&dV, H.size() * 2)); CUDA_CHECK(cudaMalloc(&dO, H.size() * 2));
            CUDA_CHECK(cudaMalloc(&dS, (size_t)BH * N * N * 2));
            fill_normal(H, 1); CUDA_CHECK(cudaMemcpy(dQ, H.data(), H.size()*2, cudaMemcpyHostToDevice));
            fill_normal(H, 2); CUDA_CHECK(cudaMemcpy(dK, H.data(), H.size()*2, cudaMemcpyHostToDevice));
            fill_normal(H, 3); CUDA_CHECK(cudaMemcpy(dV, H.data(), H.size()*2, cudaMemcpyHostToDevice));
            const float scale = 1.f / std::sqrt((float)d);

            std::printf("\n=== BH=%d N=%d d=%d  (score matrix %.0f MB) ===\n",
                        BH, N, d, (double)BH * N * N * 2 / 1e6);
            double ms_unf[2];
            for (int c = 0; c < 2; ++c) {
                ms_unf[c] = time_ms([&] {
                    attention_unfused(h, dQ, dK, dV, dO, dS, BH, N, d, scale, c);
                }, 3, 10);
                // TFLOPS counts only useful work: causal does half the pairs.
                const double f = 4.0 * BH * (double)N * N * d * (c ? 0.5 : 1.0);
                std::printf("  %-26s %8.3f ms  %6.2f TFLOPS\n",
                            c ? "unfused cuBLAS causal" : "unfused cuBLAS", ms_unf[c],
                            f / (ms_unf[c] * 1e-3) / 1e12);
            }
            for (const Variant& v : variants) {
                if (v.d != d) continue;
                const double ms = time_ms([&] { v.fn(dQ, dK, dV, dO, BH, N, scale, nullptr, BH, 1); }, 3, 10);
                const double f = 4.0 * BH * (double)N * N * d * (v.causal ? 0.5 : 1.0);
                std::printf("  %-26s %8.3f ms  %6.2f TFLOPS  %5.2fx unfused%s\n",
                            v.name, ms, f / (ms * 1e-3) / 1e12, ms_unf[v.causal] / ms,
                            ms < ms_unf[v.causal] ? "" : "  <-- slower");
            }
            CUDA_CHECK(cudaFree(dQ)); CUDA_CHECK(cudaFree(dK)); CUDA_CHECK(cudaFree(dV));
            CUDA_CHECK(cudaFree(dO)); CUDA_CHECK(cudaFree(dS));
        }
    }
    CUBLAS_CHECK(cublasDestroy(h));
    return 0;
}
