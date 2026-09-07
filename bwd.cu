// Phase 4 driver.
//
// Three levels of check, deliberately layered so a failure localises:
//   1. finite differences vs the fp64 analytic reference  (is the MATH right?)
//   2. the kernel vs the fp64 analytic reference          (is the KERNEL right?)
//   3. all four input regimes                             (is it right in fp16?)
//
// Level 1 matters: without it, levels 2 and 3 only prove the kernel agrees with
// whatever formula was typed into the reference.

#include "common.cuh"
#include "flash_fwd.cuh"
#include "flash_bwd.cuh"

namespace {

// ---------------------------------------------------------------------------
// fp64 analytic reference for dQ, dK, dV. Small shapes only.
// ---------------------------------------------------------------------------
void bwd_ref_f64(const std::vector<__half>& Q, const std::vector<__half>& K,
                 const std::vector<__half>& V, const std::vector<__half>& dOh,
                 std::vector<double>& dQ, std::vector<double>& dK,
                 std::vector<double>& dV, int BH, int N, int d, double scale,
                 bool causal) {
    dQ.assign((size_t)BH * N * d, 0.0);
    dK.assign(dQ.size(), 0.0);
    dV.assign(dQ.size(), 0.0);
    std::vector<double> p(N), o(d);

    for (int b = 0; b < BH; ++b) {
        const size_t base = (size_t)b * N * d;
        auto q = [&](int i, int k) { return (double)__half2float(Q  [base + (size_t)i * d + k]); };
        auto kk= [&](int j, int k) { return (double)__half2float(K  [base + (size_t)j * d + k]); };
        auto v = [&](int j, int k) { return (double)__half2float(V  [base + (size_t)j * d + k]); };
        auto g = [&](int i, int k) { return (double)__half2float(dOh[base + (size_t)i * d + k]); };

        for (int i = 0; i < N; ++i) {
            const int lim = causal ? i + 1 : N;
            double mx = -INFINITY;
            for (int j = 0; j < lim; ++j) {
                double a = 0.0;
                for (int k = 0; k < d; ++k) a += q(i, k) * kk(j, k);
                p[j] = a * scale;
                if (p[j] > mx) mx = p[j];
            }
            double sum = 0.0;
            for (int j = 0; j < lim; ++j) { p[j] = std::exp(p[j] - mx); sum += p[j]; }
            for (int j = 0; j < lim; ++j) p[j] /= sum;

            for (int k = 0; k < d; ++k) {
                o[k] = 0.0;
                for (int j = 0; j < lim; ++j) o[k] += p[j] * v(j, k);
            }
            double Dv = 0.0;
            for (int k = 0; k < d; ++k) Dv += g(i, k) * o[k];

            for (int j = 0; j < lim; ++j) {
                double dp = 0.0;
                for (int k = 0; k < d; ++k) dp += g(i, k) * v(j, k);
                const double ds = p[j] * (dp - Dv);
                for (int k = 0; k < d; ++k) {
                    dV[base + (size_t)j * d + k] += p[j] * g(i, k);
                    dQ[base + (size_t)i * d + k] += ds * scale * kk(j, k);
                    dK[base + (size_t)j * d + k] += ds * scale * q(i, k);
                }
            }
        }
    }
}

// Scalar loss = sum(dO * O), whose gradient wrt Q/K/V is exactly what
// bwd_ref_f64 computes. Finite-differencing this validates the formula itself.
double loss_f64(const std::vector<__half>& Q, const std::vector<__half>& K,
                const std::vector<__half>& V, const std::vector<__half>& dOh,
                int BH, int N, int d, double scale, bool causal,
                int pert_which, size_t pert_idx, double pert) {
    auto rd = [&](const std::vector<__half>& A, int which, size_t idx) {
        double x = (double)__half2float(A[idx]);
        return (which == pert_which && idx == pert_idx) ? x + pert : x;
    };
    double loss = 0.0;
    std::vector<double> p(N);
    for (int b = 0; b < BH; ++b) {
        const size_t base = (size_t)b * N * d;
        for (int i = 0; i < N; ++i) {
            const int lim = causal ? i + 1 : N;
            double mx = -INFINITY;
            for (int j = 0; j < lim; ++j) {
                double a = 0.0;
                for (int k = 0; k < d; ++k)
                    a += rd(Q, 0, base + (size_t)i * d + k) * rd(K, 1, base + (size_t)j * d + k);
                p[j] = a * scale;
                if (p[j] > mx) mx = p[j];
            }
            double sum = 0.0;
            for (int j = 0; j < lim; ++j) { p[j] = std::exp(p[j] - mx); sum += p[j]; }
            for (int k = 0; k < d; ++k) {
                double o = 0.0;
                for (int j = 0; j < lim; ++j) o += p[j] * rd(V, 2, base + (size_t)j * d + k);
                loss += (o / sum) * (double)__half2float(dOh[base + (size_t)i * d + k]);
            }
        }
    }
    return loss;
}

struct Regime { const char* name; float scale; };
const Regime kRegimes[] = {
    {"1 N(0,1)      ", 1.00f},
    {"2 large x10   ", 10.0f},
    {"3 near-uniform", 0.05f},
    {"4 near-one-hot", 3.00f},
};

using BwdFn = void (*)(const __half*, const __half*, const __half*, const __half*,
                       const __half*, float*, float*, float*, float*, float*,
                       int, int, float, int, int);
using FwdFn = void (*)(const __half*, const __half*, const __half*, __half*,
                       int, int, float, float*, int, int);
struct BVar { const char* name; BwdFn bwd; FwdFn fwd; bool causal; };

ErrStats cmp_rms(const std::vector<float>& got, const std::vector<double>& ref) {
    double a = 0.0;
    for (double r : ref) a += r * r;
    return compare(got, ref, std::sqrt(a / (double)ref.size()));
}

}  // namespace

// ---------------------------------------------------------------------------
int main() {
    print_device_banner();
    constexpr int D = 64, BR = 64, BC = 32, T = 128;
    const float scale = 1.f / std::sqrt((float)D);

    // ---------------- level 1: is the reference formula itself right? -------
    {
        const int BH = 1, N = 32;
        std::vector<__half> Q((size_t)N * D), K(Q.size()), V(Q.size()), G(Q.size());
        fill_normal(Q, 7); fill_normal(K, 8); fill_normal(V, 9); fill_normal(G, 10);
        std::vector<double> rQ, rK, rV;
        bwd_ref_f64(Q, K, V, G, rQ, rK, rV, BH, N, D, scale, false);

        std::printf("\n=== level 1: finite differences vs the analytic reference ===\n");
        std::mt19937 rng(4242);
        const double h = 1e-4;
        double worst = 0.0;
        for (int trial = 0; trial < 12; ++trial) {
            const int which = trial % 3;
            const size_t idx = rng() % Q.size();
            const double lp = loss_f64(Q, K, V, G, BH, N, D, scale, false, which, idx,  h);
            const double lm = loss_f64(Q, K, V, G, BH, N, D, scale, false, which, idx, -h);
            const double fd = (lp - lm) / (2 * h);
            const double an = (which == 0 ? rQ : which == 1 ? rK : rV)[idx];
            const double rel = std::fabs(fd - an) / std::max(std::fabs(an), 1e-3);
            worst = std::max(worst, rel);
        }
        std::printf("  worst relative disagreement over 12 probes: %.3e  -> %s\n",
                    worst, worst < 1e-4 ? "formula CONFIRMED" : "FORMULA IS WRONG");
    }

    // ---------------- levels 2 and 3: kernel vs reference, four regimes -----
    const BVar bvars[] = {
        {"fp32   ", flash_bwd::launch<BR,BC,D,T,false,false,0>,  flash::launch<BR,BC,D,T,false,D,true>, false},
        {"half2/16", flash_bwd::launch<BR,BC,D,T,false,false,16>, flash::launch<BR,BC,D,T,false,D,true>, false},
        {"half2/8 ", flash_bwd::launch<BR,BC,D,T,false,false,8>,  flash::launch<BR,BC,D,T,false,D,true>, false},
        {"fp32 BC64", flash_bwd::launch<64,64,D,T,false,false,0>, flash::launch<BR,BC,D,T,false,D,true>, false},
        {"h2dvdk   ", flash_bwd::launch<64,64,D,T,false,false,0,float,true>, flash::launch<BR,BC,D,T,false,D,true>, false},
        {"h2dvdk  c", flash_bwd::launch<64,64,D,T,true, false,0,float,true>, flash::launch<BR,BC,D,T,true, D,true>, true},
        {"fp32   c", flash_bwd::launch<BR,BC,D,T,true, false,0>,  flash::launch<BR,BC,D,T,true, D,true>, true},
        {"half2/16c",flash_bwd::launch<BR,BC,D,T,true, false,16>, flash::launch<BR,BC,D,T,true, D,true>, true},
    };

    std::printf("\n=== level 2/3: kernel vs fp64 (BH=2, N=256, d=64).  * = over 5e-2 ===\n");
    std::printf("  %-10s %-15s %11s %11s %11s\n", "variant", "regime",
                "dQ max_rel", "dK max_rel", "dV max_rel");
    for (const BVar& bv : bvars) {
        for (const Regime& rg : kRegimes) {
            const int BH = 2, N = 256;
            const size_t n = (size_t)BH * N * D;
            std::vector<__half> Q(n), K(n), V(n), G(n);
            fill_normal(Q, 101, rg.scale); fill_normal(K, 202, rg.scale);
            fill_normal(V, 303, 1.f);      fill_normal(G, 404, 1.f);

            __half *dQh,*dKh,*dVh,*dGh,*dOut; float *dLse,*dDv,*ddQ,*ddK,*ddV;
            CUDA_CHECK(cudaMalloc(&dQh,n*2)); CUDA_CHECK(cudaMalloc(&dKh,n*2));
            CUDA_CHECK(cudaMalloc(&dVh,n*2)); CUDA_CHECK(cudaMalloc(&dGh,n*2));
            CUDA_CHECK(cudaMalloc(&dOut,n*2));
            CUDA_CHECK(cudaMalloc(&dLse,(size_t)BH*N*4)); CUDA_CHECK(cudaMalloc(&dDv,(size_t)BH*N*4));
            CUDA_CHECK(cudaMalloc(&ddQ,n*4)); CUDA_CHECK(cudaMalloc(&ddK,n*4)); CUDA_CHECK(cudaMalloc(&ddV,n*4));
            CUDA_CHECK(cudaMemcpy(dQh,Q.data(),n*2,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dKh,K.data(),n*2,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dVh,V.data(),n*2,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dGh,G.data(),n*2,cudaMemcpyHostToDevice));

            bv.fwd(dQh, dKh, dVh, dOut, BH, N, scale, dLse, BH, 1);
            CUDA_CHECK(cudaDeviceSynchronize());
            bv.bwd(dQh, dKh, dVh, dGh, dOut, dLse, dDv, ddQ, ddK, ddV, BH, N, scale, BH, 1);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> hQ(n), hK(n), hV(n);
            CUDA_CHECK(cudaMemcpy(hQ.data(),ddQ,n*4,cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hK.data(),ddK,n*4,cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hV.data(),ddV,n*4,cudaMemcpyDeviceToHost));
            std::vector<double> rQ,rK,rV;
            bwd_ref_f64(Q,K,V,G,rQ,rK,rV,BH,N,D,scale,bv.causal);
            ErrStats eq=cmp_rms(hQ,rQ), ek=cmp_rms(hK,rK), ev=cmp_rms(hV,rV);
            std::printf("  %-10s %-15s %10.2e%s %10.2e%s %10.2e%s\n", bv.name, rg.name,
                        eq.max_rel, eq.max_rel<5e-2?" ":"*",
                        ek.max_rel, ek.max_rel<5e-2?" ":"*",
                        ev.max_rel, ev.max_rel<5e-2?" ":"*");
            cudaFree(dQh);cudaFree(dKh);cudaFree(dVh);cudaFree(dGh);cudaFree(dOut);
            cudaFree(dLse);cudaFree(dDv);cudaFree(ddQ);cudaFree(ddK);cudaFree(ddV);
        }
    }

    // ---------------- speed ----------------
    const int BH = 16;
    for (int N : {2048, 4096}) {
        const size_t n = (size_t)BH * N * D;
        __half *dQh,*dKh,*dVh,*dGh,*dOut; float *dLse,*dDv,*ddQ,*ddK,*ddV;
        CUDA_CHECK(cudaMalloc(&dQh,n*2)); CUDA_CHECK(cudaMalloc(&dKh,n*2));
        CUDA_CHECK(cudaMalloc(&dVh,n*2)); CUDA_CHECK(cudaMalloc(&dGh,n*2));
        CUDA_CHECK(cudaMalloc(&dOut,n*2));
        CUDA_CHECK(cudaMalloc(&dLse,(size_t)BH*N*4)); CUDA_CHECK(cudaMalloc(&dDv,(size_t)BH*N*4));
        CUDA_CHECK(cudaMalloc(&ddQ,n*4)); CUDA_CHECK(cudaMalloc(&ddK,n*4)); CUDA_CHECK(cudaMalloc(&ddV,n*4));
        std::vector<__half> H(n);
        fill_normal(H,1); CUDA_CHECK(cudaMemcpy(dQh,H.data(),n*2,cudaMemcpyHostToDevice));
        fill_normal(H,2); CUDA_CHECK(cudaMemcpy(dKh,H.data(),n*2,cudaMemcpyHostToDevice));
        fill_normal(H,3); CUDA_CHECK(cudaMemcpy(dVh,H.data(),n*2,cudaMemcpyHostToDevice));
        fill_normal(H,4); CUDA_CHECK(cudaMemcpy(dGh,H.data(),n*2,cudaMemcpyHostToDevice));

        std::printf("\n=== speed  BH=%d N=%d d=%d ===\n", BH, N, D);
        const double fwd_nc = time_ms([&]{ flash::launch<BR,BC,D,T,false,16,false>(dQh,dKh,dVh,dOut,BH,N,scale,dLse,BH,1); },3,10);
        const double fwd_c  = time_ms([&]{ flash::launch<BR,BC,D,T,true, 16,false>(dQh,dKh,dVh,dOut,BH,N,scale,dLse,BH,1); },3,10);
        std::printf("  %-12s causal=0 %8.3f ms (%5.2f TF)\n", "forward", fwd_nc,
                    4.0*BH*(double)N*N*D/(fwd_nc*1e-3)/1e12);
        std::printf("  %-12s causal=1 %8.3f ms (%5.2f TF)\n", "forward", fwd_c,
                    2.0*BH*(double)N*N*D/(fwd_c*1e-3)/1e12);
        for (const BVar& bv : bvars) {
            const double h = bv.causal ? 0.5 : 1.0;
            const double ms = time_ms([&]{ bv.bwd(dQh,dKh,dVh,dGh,dOut,dLse,dDv,ddQ,ddK,ddV,BH,N,scale,BH,1); },3,10);
            std::printf("  bwd %-9s causal=%d %8.3f ms (%5.2f TF)  bwd/fwd %.2fx\n",
                        bv.name, (int)bv.causal, ms,
                        10.0*BH*(double)N*N*D*h/(ms*1e-3)/1e12,
                        ms / (bv.causal ? fwd_c : fwd_nc));
        }
        cudaFree(dQh);cudaFree(dKh);cudaFree(dVh);cudaFree(dGh);cudaFree(dOut);
        cudaFree(dLse);cudaFree(dDv);cudaFree(ddQ);cudaFree(ddK);cudaFree(ddV);
    }
    return 0;
}
