// Phase 5 — torch binding for the sm_60 FlashAttention kernels.
//
// Tensors arrive as (B, H, N, D) fp16 and the kernels want a flat (B*H, N, D),
// which a contiguous 4-D tensor already is, so no repacking happens here.
//
// Only the FORWARD exposes a precision choice. The backward is fp32 throughout
// and deliberately has no fp16 tier: measured at 2026-09-06, half2 MACs bought
// 13% and blew regimes 2 and 4 out to ~1e0, because dS = P*(dP - D) is a
// cancellation. See PLAN.md Phase 4.

#include <torch/extension.h>
#include <ATen/cuda/Exceptions.h>   // AT_CUDA_CHECK
#include "flash_fwd.cuh"
#include "flash_bwd.cuh"

namespace {
// Tile shapes differ per head dim because shared memory is the binding
// constraint, and the forward and backward hit it at different points.
//   d=64  fwd 64x64 (24 KB)   bwd 64x64 (48 KB, at the per-block cap)
//   d=128 fwd 64x32 (32 KB)   bwd 32x32 (36 KB) -- 64x32 would need 56 KB
template <int D> struct Tiles;
template <> struct Tiles<64>  {
    static constexpr int FBR = 64, FBC = 64, FT = 128;
    static constexpr int BBR = 64, BBC = 64, BT = 128;
};
template <> struct Tiles<128> {
    static constexpr int FBR = 64, FBC = 32, FT = 128;
    static constexpr int BBR = 32, BBC = 32, BT = 128;
};

void check_in(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == at::kHalf, name, " must be fp16");
    TORCH_CHECK(t.dim() == 4, name, " must be (B, H, N, D)");
}
}  // namespace

template <int D>
void fwd_dispatch(const __half* q, const __half* k, const __half* v, __half* o,
                  float* lse, int BH, int N, float scale, int Hq, int GROUP,
                  bool causal, bool full_precision) {
    using T = Tiles<D>;
    constexpr int BR = T::FBR, BC = T::FBC, TH = T::FT;
    if (causal) {
        if (full_precision) flash::launch<BR,BC,D,TH,true,  D,  true >(q,k,v,o,BH,N,scale,lse,Hq,GROUP);
        else                flash::launch<BR,BC,D,TH,true,  16, false>(q,k,v,o,BH,N,scale,lse,Hq,GROUP);
    } else {
        if (full_precision) flash::launch<BR,BC,D,TH,false, D,  true >(q,k,v,o,BH,N,scale,lse,Hq,GROUP);
        else                flash::launch<BR,BC,D,TH,false, 16, false>(q,k,v,o,BH,N,scale,lse,Hq,GROUP);
    }
}

std::vector<at::Tensor> fa_forward(at::Tensor q, at::Tensor k, at::Tensor v,
                                   bool causal, bool full_precision) {
    check_in(q, "q"); check_in(k, "k"); check_in(v, "v");
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous();
    const int B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);
    const int Hkv = k.size(1);
    TORCH_CHECK(D == 64 || D == 128, "head dim must be 64 or 128, got ", D);
    TORCH_CHECK(N > 0, "N must be positive");
    TORCH_CHECK(Hkv > 0 && H % Hkv == 0,
                "query heads (", H, ") must be a multiple of key/value heads (", Hkv, ")");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must match");
    TORCH_CHECK(k.size(0) == B && k.size(2) == N && k.size(3) == D,
                "k/v must be (B, Hkv, N, D) matching q's B, N, D");
    const int GROUP = H / Hkv;          // 1 = MHA, H = MQA, else GQA

    auto o   = at::empty_like(q);
    auto lse = at::empty({B, H, N}, q.options().dtype(at::kFloat));
    const int BH = B * H;
    const float scale = 1.f / std::sqrt((float)D);

    auto* pq = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    auto* pk = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
    auto* pv = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
    auto* po = reinterpret_cast<__half*>(o.data_ptr<at::Half>());
    auto* pl = lse.data_ptr<float>();

    if (D == 64) fwd_dispatch<64 >(pq,pk,pv,po,pl,BH,N,scale,H,GROUP,causal,full_precision);
    else         fwd_dispatch<128>(pq,pk,pv,po,pl,BH,N,scale,H,GROUP,causal,full_precision);
    AT_CUDA_CHECK(cudaGetLastError());
    return {o, lse};
}

template <int D>
void bwd_dispatch(const __half* q, const __half* k, const __half* v,
                  const __half* dO, const __half* o, float* lse, float* dvec,
                  float* dq, __half* dk, __half* dv, int BH, int N, float scale,
                  int Hq, int GROUP, bool causal) {
    using T = Tiles<D>;
    constexpr int BR = T::BBR, BC = T::BBC, TH = T::BT;
    // H2_DVDK (last param): __hfma2 in the dV/dK contraction only. Measured
    // +8% non-causal / +13% causal with all four input regimes still clean --
    // unlike half2 in the OUTER products, which is +13% and wrecks them.
    if (causal)
        flash_bwd::launch<BR,BC,D,TH,true, false,0,__half,true>(
            q,k,v,dO,o,lse,dvec,dq,dk,dv,BH,N,scale,Hq,GROUP);
    else
        flash_bwd::launch<BR,BC,D,TH,false,false,0,__half,true>(
            q,k,v,dO,o,lse,dvec,dq,dk,dv,BH,N,scale,Hq,GROUP);
}

std::vector<at::Tensor> fa_backward(at::Tensor q, at::Tensor k, at::Tensor v,
                                    at::Tensor dO, at::Tensor o, at::Tensor lse,
                                    bool causal) {
    check_in(q, "q"); check_in(dO, "dO");
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous();
    dO = dO.contiguous(); o = o.contiguous(); lse = lse.contiguous();
    const int B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);
    const int Hkv = k.size(1);
    TORCH_CHECK(D == 64 || D == 128, "head dim must be 64 or 128, got ", D);
    TORCH_CHECK(N > 0, "N must be positive");
    TORCH_CHECK(Hkv > 0 && H % Hkv == 0, "query heads must be a multiple of kv heads");
    const int GROUP = H / Hkv;

    const int BH = B * H;
    const float scale = 1.f / std::sqrt((float)D);
    auto f32 = q.options().dtype(at::kFloat);
    // dQ is fp32 because it is accumulated with atomicAdd across blocks.
    // dK/dV are written once from registers, so they go straight out as fp16.
    auto dq   = at::empty({B, H, N, D}, f32);
    auto dk   = at::empty_like(k);        // (B, Hkv, N, D)
    auto dv   = at::empty_like(k);
    auto dvec = at::empty({B, H, N}, f32);   // rowsum(dO * O) workspace

    auto* pq  = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    auto* pk  = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
    auto* pv  = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
    auto* pdo = reinterpret_cast<const __half*>(dO.data_ptr<at::Half>());
    auto* po  = reinterpret_cast<const __half*>(o.data_ptr<at::Half>());
    auto* pdk = reinterpret_cast<__half*>(dk.data_ptr<at::Half>());
    auto* pdv = reinterpret_cast<__half*>(dv.data_ptr<at::Half>());

    if (D == 64)
        bwd_dispatch<64 >(pq,pk,pv,pdo,po, lse.data_ptr<float>(), dvec.data_ptr<float>(),
                          dq.data_ptr<float>(), pdk, pdv, BH, N, scale, H, GROUP, causal);
    else
        bwd_dispatch<128>(pq,pk,pv,pdo,po, lse.data_ptr<float>(), dvec.data_ptr<float>(),
                          dq.data_ptr<float>(), pdk, pdv, BH, N, scale, H, GROUP, causal);
    AT_CUDA_CHECK(cudaGetLastError());
    return {dq, dk, dv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &fa_forward,  "FlashAttention forward (sm_60)");
    m.def("backward", &fa_backward, "FlashAttention backward (sm_60)");
}
