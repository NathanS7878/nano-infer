// Fused RMSNorm — nano-infer Phase 3, kernel 1.
//
// PyTorch runs RMSNorm as five separate kernels, each making a full round trip
// through VRAM:
//     x.pow(2)        read x,   write [b,seq,hidden]
//     .mean(-1)       read it,  write [b,seq,1]
//     rsqrt(v+eps)    read,     write
//     x * scale       read x + scale, write [b,seq,hidden]
//     weight * that   read,     write [b,seq,hidden]
//
// The arithmetic is trivial (~1 FLOP per byte moved, against this card's
// ~364 FLOP/byte roofline ridge), so all of that is memory traffic spent on
// nothing. This kernel does the whole thing in ONE trip: read the row once,
// reduce and scale in registers, write once.
//
// Structure: one thread block per row (one token's `hidden` values).
//   1. each thread accumulates a partial sum of squares over a strided slice
//      (strided so consecutive threads touch consecutive addresses — coalesced)
//   2. warp-level reduction via __shfl_down_sync: threads trade values directly
//      through registers, no shared memory, no global memory
//   3. one value per warp lands in shared memory; warp 0 reduces those
//   4. broadcast rsqrt(mean + eps) and write the normalized, weighted row

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define WARP_SIZE 32
#define MAX_WARPS 32

__inline__ __device__ float warp_reduce_sum(float val) {
    // Butterfly reduction: halve the number of contributing lanes each round.
    // __shfl_down_sync reads another lane's register directly — this is why a
    // warp reduction costs essentially nothing compared to a memory round trip.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void rmsnorm_kernel(
        const scalar_t* __restrict__ x,
        const scalar_t* __restrict__ weight,
        scalar_t* __restrict__ out,
        const int hidden,
        const float eps) {

    const long long row = blockIdx.x;
    const scalar_t* __restrict__ x_row = x + row * hidden;
    scalar_t* __restrict__ out_row = out + row * hidden;

    // --- 1. partial sum of squares, accumulated in fp32 for stability ---
    float partial = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = static_cast<float>(x_row[i]);
        partial += v * v;
    }

    // --- 2. reduce within each warp ---
    partial = warp_reduce_sum(partial);

    // --- 3. reduce across warps through shared memory ---
    __shared__ float s_warp[MAX_WARPS];
    __shared__ float s_scale;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    if (lane == 0) s_warp[warp] = partial;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < num_warps) ? s_warp[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            s_scale = rsqrtf(v / static_cast<float>(hidden) + eps);
        }
    }
    __syncthreads();
    const float scale = s_scale;

    // --- 4. normalize, apply the learned weight, write once ---
    // Cast order mirrors the PyTorch reference exactly: normalize in fp32, cast
    // back to the input dtype, THEN multiply by the weight. Doing the weight
    // multiply in fp32 instead would round differently.
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const scalar_t normed = static_cast<scalar_t>(static_cast<float>(x_row[i]) * scale);
        out_row[i] = normed * weight[i];
    }
}

// ---------------------------------------------------------------------------
// Vectorized variant.
//
// The scalar kernel above loads one 2-byte half per thread per iteration. The
// memory system moves data in far wider transactions than that, so each thread
// spends most of its time with only 2 bytes in flight. Loading float4 (16 bytes
// = 8 halves) per thread raises memory-level parallelism by 8x for the same
// arithmetic, which is exactly the lever for a bandwidth-bound kernel.
//
// Requires hidden % VEC == 0 and 16-byte-aligned rows; the host function checks
// both and falls back to the scalar kernel otherwise.
// ---------------------------------------------------------------------------

template <typename scalar_t, int VEC>
__global__ void rmsnorm_vec_kernel(
        const scalar_t* __restrict__ x,
        const scalar_t* __restrict__ weight,
        scalar_t* __restrict__ out,
        const int hidden,
        const float eps) {

    using Vec = float4;                       // 16 bytes, the widest single load
    const long long row = blockIdx.x;
    const int nvec = hidden / VEC;

    const Vec* __restrict__ x_vec =
        reinterpret_cast<const Vec*>(x + row * hidden);
    const Vec* __restrict__ w_vec = reinterpret_cast<const Vec*>(weight);
    Vec* __restrict__ out_vec = reinterpret_cast<Vec*>(out + row * hidden);

    // --- 1. sum of squares over vector loads ---
    float partial = 0.0f;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        Vec chunk = x_vec[i];
        const scalar_t* vals = reinterpret_cast<const scalar_t*>(&chunk);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float v = static_cast<float>(vals[j]);
            partial += v * v;
        }
    }

    // --- 2/3. same two-stage reduction as the scalar kernel ---
    partial = warp_reduce_sum(partial);

    __shared__ float s_warp[MAX_WARPS];
    __shared__ float s_scale;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    if (lane == 0) s_warp[warp] = partial;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < num_warps) ? s_warp[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s_scale = rsqrtf(v / static_cast<float>(hidden) + eps);
    }
    __syncthreads();
    const float scale = s_scale;

    // --- 4. scale, weight, store — one wide write per 8 elements ---
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        Vec chunk = x_vec[i];
        Vec wchunk = w_vec[i];
        scalar_t* vals = reinterpret_cast<scalar_t*>(&chunk);
        const scalar_t* wv = reinterpret_cast<const scalar_t*>(&wchunk);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const scalar_t normed =
                static_cast<scalar_t>(static_cast<float>(vals[j]) * scale);
            vals[j] = normed * wv[j];
        }
        out_vec[i] = chunk;
    }
}

static inline bool is_aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "dtype mismatch");

    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();

    const int hidden = static_cast<int>(x_c.size(-1));
    TORCH_CHECK(w_c.numel() == hidden, "weight size must equal x.size(-1)");
    const long long rows = x_c.numel() / hidden;

    auto out = torch::empty_like(x_c);
    if (rows == 0) return out;

    // 256 threads: enough to saturate memory on a 896-wide row without leaving
    // most of the block idle on shorter rows.
    int threads = 256;
    if (hidden < threads) {
        threads = ((hidden + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        if (threads < WARP_SIZE) threads = WARP_SIZE;
    }

    const at::cuda::OptionalCUDAGuard guard(device_of(x_c));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x_c.scalar_type(), "rmsnorm_forward", [&] {
            const scalar_t* xp = x_c.data_ptr<scalar_t>();
            const scalar_t* wp = w_c.data_ptr<scalar_t>();
            scalar_t* op = out.data_ptr<scalar_t>();

            // 16 bytes per vector load -> 8 halves, or 4 floats.
            constexpr int VEC = 16 / sizeof(scalar_t);
            const bool vectorizable =
                (hidden % VEC == 0) &&
                ((static_cast<size_t>(hidden) * sizeof(scalar_t)) % 16 == 0) &&
                is_aligned16(xp) && is_aligned16(wp) && is_aligned16(op);

            if (vectorizable) {
                rmsnorm_vec_kernel<scalar_t, VEC>
                    <<<static_cast<int>(rows), threads, 0, stream>>>(
                        xp, wp, op, hidden, static_cast<float>(eps));
            } else {
                rmsnorm_kernel<scalar_t>
                    <<<static_cast<int>(rows), threads, 0, stream>>>(
                        xp, wp, op, hidden, static_cast<float>(eps));
            }
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward,
          "Fused RMSNorm (CUDA)",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("eps"));
}
