// Fused SwiGLU — nano-infer Phase 3, kernel 2.
//
// The op is the gated valve in the middle of the MLP:
//
//     hidden = silu(gate) * up          silu(z) = z * sigmoid(z) = z / (1 + e^-z)
//
// PyTorch runs this as two kernels over [rows, 4864] fp16 tensors:
//
//     F.silu(gate)     read gate (2B)             write tmp  (2B)   = 4 B/elem
//     tmp * up         read tmp (2B) + up (2B)    write out  (2B)   = 6 B/elem
//     ------------------------------------------------------------------------
//     total                                                          10 B/elem
//
// The compulsory minimum is 6 B/elem: read gate, read up, write out. The whole
// win available here is those 4 wasted bytes — the `tmp` round trip through
// VRAM. That caps the honest speedup at 10/6 = 1.67x, which is FAR less than
// RMSNorm's 7.66x, and for a good reason: RMSNorm was five round trips
// collapsed into one, this is two collapsed into one.
//
// This is the kernel that teaches the real lesson of the phase: the win is not
// "CUDA is faster than PyTorch." The win is exactly the memory traffic removed,
// and it can be predicted to within a few percent BEFORE writing any code.
//
// NUMERICS — this must mirror PyTorch op-for-op or parity fails.
// ATen's silu on half promotes to float, divides in float, and casts back:
//     opmath_t xa = float(x);  return xa / (1 + exp(-xa));
// The multiply by `up` then happens in half. Doing the multiply in float and
// casting once at the end would be MORE accurate and WRONG for our purposes:
// it would round differently from the reference. Match, do not improve.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Mirrors at::native::silu's fp16 path exactly: promote, divide in fp32, cast back.
template <typename scalar_t>
__inline__ __device__ scalar_t silu_mul(scalar_t g, scalar_t u) {
    const float gf = static_cast<float>(g);
    const scalar_t s = static_cast<scalar_t>(gf / (1.0f + expf(-gf)));
    return s * u;
}

template <typename scalar_t>
__global__ void swiglu_kernel(
        const scalar_t* __restrict__ gate,
        const scalar_t* __restrict__ up,
        scalar_t* __restrict__ out,
        const long long n) {

    // Grid-stride loop: the grid is sized to the GPU, not to the data, so one
    // launch configuration covers every shape from a single decode row to a
    // 32-sequence prefill without relaunching or overflowing gridDim.
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)gridDim.x * blockDim.x) {
        out[i] = silu_mul<scalar_t>(gate[i], up[i]);
    }
}

// ---------------------------------------------------------------------------
// Vectorized variant.
//
// Same lesson as RMSNorm kernel 1: a thread holding one 2-byte half has almost
// nothing in flight, and a bandwidth-bound kernel lives or dies on memory-level
// parallelism. float4 = 16 bytes = 8 halves per load, 8x the bytes in flight
// for the same instruction count.
//
// 4864 % 8 == 0, so every shape the model actually produces takes this path.
// ---------------------------------------------------------------------------

template <typename scalar_t, int VEC>
__global__ void swiglu_vec_kernel(
        const scalar_t* __restrict__ gate,
        const scalar_t* __restrict__ up,
        scalar_t* __restrict__ out,
        const long long nvec) {

    using Vec = float4;
    const Vec* __restrict__ g_vec = reinterpret_cast<const Vec*>(gate);
    const Vec* __restrict__ u_vec = reinterpret_cast<const Vec*>(up);
    Vec* __restrict__ o_vec = reinterpret_cast<Vec*>(out);

    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < nvec; i += (long long)gridDim.x * blockDim.x) {
        Vec gchunk = g_vec[i];
        Vec uchunk = u_vec[i];
        scalar_t* gv = reinterpret_cast<scalar_t*>(&gchunk);
        const scalar_t* uv = reinterpret_cast<const scalar_t*>(&uchunk);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            gv[j] = silu_mul<scalar_t>(gv[j], uv[j]);
        }
        o_vec[i] = gchunk;
    }
}

static inline bool swiglu_is_aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

torch::Tensor swiglu_forward(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
    TORCH_CHECK(up.is_cuda(), "up must be a CUDA tensor");
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "dtype mismatch");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have the same shape");

    auto g_c = gate.contiguous();
    auto u_c = up.contiguous();
    auto out = torch::empty_like(g_c);

    const long long n = g_c.numel();
    if (n == 0) return out;

    const int threads = 256;

    const at::cuda::OptionalCUDAGuard guard(device_of(g_c));
    auto stream = at::cuda::getCurrentCUDAStream();

    // Cap the grid at a few waves over the device rather than one block per
    // element: past saturation, extra blocks only add scheduling overhead.
    const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int max_blocks = sm_count * 32;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        g_c.scalar_type(), "swiglu_forward", [&] {
            const scalar_t* gp = g_c.data_ptr<scalar_t>();
            const scalar_t* upp = u_c.data_ptr<scalar_t>();
            scalar_t* op = out.data_ptr<scalar_t>();

            constexpr int VEC = 16 / sizeof(scalar_t);
            const bool vectorizable =
                (n % VEC == 0) &&
                swiglu_is_aligned16(gp) && swiglu_is_aligned16(upp) &&
                swiglu_is_aligned16(op);

            if (vectorizable) {
                const long long nvec = n / VEC;
                long long want = (nvec + threads - 1) / threads;
                const int blocks = static_cast<int>(want < max_blocks ? want : max_blocks);
                swiglu_vec_kernel<scalar_t, VEC>
                    <<<blocks, threads, 0, stream>>>(gp, upp, op, nvec);
            } else {
                long long want = (n + threads - 1) / threads;
                const int blocks = static_cast<int>(want < max_blocks ? want : max_blocks);
                swiglu_kernel<scalar_t>
                    <<<blocks, threads, 0, stream>>>(gp, upp, op, n);
            }
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
