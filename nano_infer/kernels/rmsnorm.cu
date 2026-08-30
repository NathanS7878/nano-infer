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
            rmsnorm_kernel<scalar_t><<<static_cast<int>(rows), threads, 0, stream>>>(
                x_c.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                hidden,
                static_cast<float>(eps));
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward,
          "Fused RMSNorm (CUDA)",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("eps"));
}
