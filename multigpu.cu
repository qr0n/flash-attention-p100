// Phase 6 — two-GPU check. The plan's guidance is "data parallel and nothing
// fancier", so the only question worth asking is whether two cards running the
// same kernel on their own shards actually deliver 2x, or whether the PCIe /
// NUMA topology gets in the way. There is no inter-card traffic in a
// data-parallel forward, so anything short of ~2x is a host-side artifact.

#include <chrono>
#include "common.cuh"
#include "flash_fwd.cuh"

int main() {
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    std::printf("devices visible: %d\n", ndev);
    if (ndev < 2) { std::printf("need two GPUs\n"); return 0; }

    constexpr int D = 64, BR = 64, BC = 64, T = 128;
    const int BH = 16, N = 4096;
    const float scale = 1.f / std::sqrt((float)D);
    const size_t n = (size_t)BH * N * D;
    const double flops = 4.0 * BH * (double)N * N * D;

    __half *q[2], *k[2], *v[2], *o[2];
    cudaStream_t st[2];
    std::vector<__half> H(n);
    for (int d = 0; d < 2; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamCreate(&st[d]));
        CUDA_CHECK(cudaMalloc(&q[d], n*2)); CUDA_CHECK(cudaMalloc(&k[d], n*2));
        CUDA_CHECK(cudaMalloc(&v[d], n*2)); CUDA_CHECK(cudaMalloc(&o[d], n*2));
        fill_normal(H, 1 + d); CUDA_CHECK(cudaMemcpy(q[d], H.data(), n*2, cudaMemcpyHostToDevice));
        fill_normal(H, 3 + d); CUDA_CHECK(cudaMemcpy(k[d], H.data(), n*2, cudaMemcpyHostToDevice));
        fill_normal(H, 5 + d); CUDA_CHECK(cudaMemcpy(v[d], H.data(), n*2, cudaMemcpyHostToDevice));
    }

    auto run_on = [&](int d, int iters) {
        CUDA_CHECK(cudaSetDevice(d));
        for (int i = 0; i < iters; ++i)
            flash::fwd_kernel<BR, BC, D, T, false, 16, false>
                <<<dim3(N/BR, BH), T, 0, st[d]>>>(q[d], k[d], v[d], o[d], nullptr, N, scale, BH, 1);
    };

    const int iters = 10;
    double single = 0.0;
    for (int d = 0; d < 2; ++d) {
        run_on(d, 3); CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaStreamSynchronize(st[d]));
        auto t0 = std::chrono::high_resolution_clock::now();
        run_on(d, iters);
        CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaStreamSynchronize(st[d]));
        auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        std::printf("  card %d alone: %7.3f ms  %5.2f TFLOPS\n", d, ms, flops/(ms*1e-3)/1e12);
        if (d == 0) single = ms;
    }

    for (int d = 0; d < 2; ++d) run_on(d, 3);
    for (int d = 0; d < 2; ++d) { CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaStreamSynchronize(st[d])); }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int d = 0; d < 2; ++d) run_on(d, iters);
    for (int d = 0; d < 2; ++d) { CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaStreamSynchronize(st[d])); }
    auto t1 = std::chrono::high_resolution_clock::now();
    const double both = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    std::printf("  both concurrent: %7.3f ms for 2x the work  %5.2f TFLOPS aggregate\n",
                both, 2*flops/(both*1e-3)/1e12);
    std::printf("  scaling: %.2fx of ideal 2.00x\n", 2.0 * single / both);
    return 0;
}
