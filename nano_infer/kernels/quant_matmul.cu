// Fused dequantize-and-matmul — nano-infer Phase 4.
//
// y[b, n] = sum_k x[b, k] * W[n, k],  with W stored quantized and NEVER
// materialized in fp16 anywhere in global memory.
//
// That last clause is the whole point, and it is easy to get wrong. Writing a
// dequantized copy of W to VRAM and calling cuBLAS on it would be far simpler,
// would give the right answer, and would move MORE bytes than plain fp16 (you
// would read the quantized weights, write an fp16 copy, then read it back).
// Quantization only pays if the unpacking happens in registers, between the
// load and the multiply. So the packed bytes are the only form of W that ever
// crosses the memory bus.
//
// WHY THIS SHOULD WIN, AND EXACTLY WHERE
// --------------------------------------
// Phase 1 measured forward cost flat at ~38-40 ms from sequence length 32 to
// 512, and Phase 2 measured decode flat at ~38 ms/step across a 32x batch
// range. Both are the signature of weight streaming: the time is spent reading
// 988 MB of weights, not doing arithmetic.
//
// Arithmetic intensity of this op, per weight:
//     fp16:  2 FLOPs / 2 bytes    = 1 FLOP/byte
//     INT8:  2 FLOPs / 1 byte     = 2 FLOP/byte
//     INT4:  2 FLOPs / 0.5 bytes  = 4 FLOP/byte
// against this card's ~364 FLOP/byte roofline ridge. All three are deeply
// memory-bound at DECODE, so bytes moved is time and the ceilings are 2x and
// 4x on the quantized layers.
//
// At PREFILL the answer flips. With a 32-sequence batch of 32-token prompts the
// weight row is reused across ~1024 rows of x, so intensity rises by that reuse
// factor and the op stops being weight-bound. Quantization buys little there
// and the unpacking is pure added work. This kernel is therefore a DECODE
// optimization, and the benchmark reports both regimes rather than the
// flattering one.
//
// PARALLELIZATION: ONE WARP PER OUTPUT ROW
// ----------------------------------------
// The obvious mapping — one block per (batch row, output row) — re-reads the
// entire weight matrix once per batch row. At batch 32 that is 32x the
// compulsory weight traffic, which would destroy the 4x saving before it
// started.
//
// So a warp owns one output row n, holds BT batch accumulators in REGISTERS,
// and streams W[n, :] exactly once while serving all BT rows of x. BT is a
// compile-time constant because a runtime-bounded loop over a local array
// spills it to local memory, which is DRAM wearing a costume.
//
// Coalescing: lane L takes chunk L, so a warp reads 32 consecutive uint32 of
// packed weights (128 contiguous bytes) and 32 consecutive 16-byte spans of x
// (512 contiguous bytes). Both fully utilised.
//
// x is re-read once per output row, relying on L2: for hidden 896 and batch 32
// that is 57 KB against a 4 MB L2, so it stays resident. The weights are the
// stream; x is the working set.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define WARP_SZ 32

__inline__ __device__ float warp_reduce_add(float v) {
    #pragma unroll
    for (int off = WARP_SZ / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// ---------------------------------------------------------------------------
// INT4, group-wise asymmetric.  packed[n, k/2]: low nibble = even k.
// ---------------------------------------------------------------------------

template <typename scalar_t, int BT>
__global__ void int4_matmul_kernel(
        const scalar_t* __restrict__ x,        // [B, K]
        const uint8_t*  __restrict__ packed,   // [N, K/2]
        const scalar_t* __restrict__ scale,    // [N, K/G]
        const uint8_t*  __restrict__ zero,     // [N, K/G]
        scalar_t* __restrict__ y,              // [B, N]
        const int B, const int N, const int K, const int G) {

    const int warps = blockDim.x / WARP_SZ;
    const int warp = threadIdx.x / WARP_SZ;
    const int lane = threadIdx.x % WARP_SZ;
    const int n = blockIdx.x * warps + warp;
    const int b0 = blockIdx.y * BT;
    if (n >= N) return;

    const int groups = K / G;
    const int nchunk = K / 8;                       // 8 nibbles per uint32
    const uint32_t* __restrict__ prow =
        reinterpret_cast<const uint32_t*>(packed + (size_t)n * (K / 2));

    float acc[BT];
    #pragma unroll
    for (int i = 0; i < BT; ++i) acc[i] = 0.0f;

    for (int c = lane; c < nchunk; c += WARP_SZ) {
        const uint32_t bits = prow[c];
        const int k0 = c * 8;
        const int g = k0 / G;
        const float s = static_cast<float>(scale[(size_t)n * groups + g]);
        const float z = static_cast<float>(zero[(size_t)n * groups + g]);

        // Unpack in REGISTERS. This is the line the whole design exists for.
        float w[8];
        #pragma unroll
        for (int t = 0; t < 8; ++t)
            w[t] = (static_cast<float>((bits >> (4 * t)) & 0xFu) - z) * s;

        #pragma unroll
        for (int bb = 0; bb < BT; ++bb) {
            const int b = b0 + bb;
            if (b < B) {
                const scalar_t* __restrict__ xr = x + (size_t)b * K + k0;
                float sum = 0.0f;
                #pragma unroll
                for (int t = 0; t < 8; ++t)
                    sum += static_cast<float>(xr[t]) * w[t];
                acc[bb] += sum;
            }
        }
    }

    #pragma unroll
    for (int bb = 0; bb < BT; ++bb) {
        const float v = warp_reduce_add(acc[bb]);
        const int b = b0 + bb;
        if (lane == 0 && b < B)
            y[(size_t)b * N + n] = static_cast<scalar_t>(v);
    }
}

// ---------------------------------------------------------------------------
// INT8, per output row, symmetric.  One scale for the whole row.
// ---------------------------------------------------------------------------

template <typename scalar_t, int BT>
__global__ void int8_matmul_kernel(
        const scalar_t* __restrict__ x,        // [B, K]
        const int8_t*   __restrict__ q,        // [N, K]
        const scalar_t* __restrict__ scale,    // [N, 1]
        scalar_t* __restrict__ y,              // [B, N]
        const int B, const int N, const int K) {

    const int warps = blockDim.x / WARP_SZ;
    const int warp = threadIdx.x / WARP_SZ;
    const int lane = threadIdx.x % WARP_SZ;
    const int n = blockIdx.x * warps + warp;
    const int b0 = blockIdx.y * BT;
    if (n >= N) return;

    const int nchunk = K / 4;                       // 4 int8 per uint32
    const uint32_t* __restrict__ qrow =
        reinterpret_cast<const uint32_t*>(q + (size_t)n * K);
    const float s = static_cast<float>(scale[n]);

    float acc[BT];
    #pragma unroll
    for (int i = 0; i < BT; ++i) acc[i] = 0.0f;

    for (int c = lane; c < nchunk; c += WARP_SZ) {
        const uint32_t bits = qrow[c];
        const int k0 = c * 4;

        // Sign-extend four int8 out of the word, in registers.
        float w[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t)
            w[t] = static_cast<float>(
                static_cast<int8_t>((bits >> (8 * t)) & 0xFFu)) * s;

        #pragma unroll
        for (int bb = 0; bb < BT; ++bb) {
            const int b = b0 + bb;
            if (b < B) {
                const scalar_t* __restrict__ xr = x + (size_t)b * K + k0;
                float sum = 0.0f;
                #pragma unroll
                for (int t = 0; t < 4; ++t)
                    sum += static_cast<float>(xr[t]) * w[t];
                acc[bb] += sum;
            }
        }
    }

    #pragma unroll
    for (int bb = 0; bb < BT; ++bb) {
        const float v = warp_reduce_add(acc[bb]);
        const int b = b0 + bb;
        if (lane == 0 && b < B)
            y[(size_t)b * N + n] = static_cast<scalar_t>(v);
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

static inline int warps_per_block(int threads) { return threads / WARP_SZ; }

#define DISPATCH_BT(BT_CONST, LAUNCH)                                          \
    case BT_CONST: { LAUNCH(BT_CONST); break; }

// Defined at FILE SCOPE. A #define inside the AT_DISPATCH lambda sits inside
// a macro expansion, which the preprocessor rejects -- nvcc reports it as a
// baffling "expected an expression" pointing at the whole expanded dispatch.

#define LAUNCH_I4(BTV)                                                         \
    int4_matmul_kernel<scalar_t, BTV><<<grid, threads, 0, stream>>>(           \
        x2.data_ptr<scalar_t>(), p_c.data_ptr<uint8_t>(),                      \
        s_c.data_ptr<scalar_t>(), z_c.data_ptr<uint8_t>(),                     \
        y.data_ptr<scalar_t>(), B, N, K, static_cast<int>(group))

#define LAUNCH_I8(BTV)                                                         \
    int8_matmul_kernel<scalar_t, BTV><<<grid, threads, 0, stream>>>(           \
        x2.data_ptr<scalar_t>(), q_c.data_ptr<int8_t>(),                       \
        s_c.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, N, K)

torch::Tensor int4_matmul(torch::Tensor x, torch::Tensor packed,
                          torch::Tensor scale, torch::Tensor zero,
                          int64_t in_features, int64_t group) {
    TORCH_CHECK(x.is_cuda() && packed.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(packed.dim() == 2, "packed must be [out, in/2]");
    TORCH_CHECK(packed.scalar_type() == at::kByte, "packed must be uint8");
    TORCH_CHECK(zero.scalar_type() == at::kByte, "zero must be uint8");

    auto x_c = x.contiguous();
    const int K = static_cast<int>(in_features);
    const int N = static_cast<int>(packed.size(0));
    TORCH_CHECK(x_c.size(-1) == K, "x last dim must equal in_features");
    TORCH_CHECK(K % 8 == 0, "in_features must be a multiple of 8");
    TORCH_CHECK(group % 8 == 0 && K % group == 0,
                "group must divide in_features and be a multiple of 8");

    auto shape = x_c.sizes().vec();
    const int B = static_cast<int>(x_c.numel() / K);
    shape.back() = N;
    auto y = torch::empty(shape, x_c.options());
    if (B == 0 || N == 0) return y;

    auto x2 = x_c.reshape({B, K});
    auto p_c = packed.contiguous();
    auto s_c = scale.contiguous();
    auto z_c = zero.contiguous();

    const int threads = 256;
    const int wpb = warps_per_block(threads);

    const at::cuda::OptionalCUDAGuard guard(device_of(x_c));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x_c.scalar_type(), "int4_matmul", [&] {
            // One batch tile covers the whole batch when it can, so the weight
            // matrix is streamed exactly once. Falling back to tiles multiplies
            // weight traffic by the tile count, which is the thing to avoid.
            int bt = 1;
            while (bt < B && bt < 32) bt <<= 1;
            const int tiles = (B + bt - 1) / bt;
            dim3 grid((N + wpb - 1) / wpb, tiles);

            switch (bt) {
                DISPATCH_BT(1, LAUNCH_I4)
                DISPATCH_BT(2, LAUNCH_I4)
                DISPATCH_BT(4, LAUNCH_I4)
                DISPATCH_BT(8, LAUNCH_I4)
                DISPATCH_BT(16, LAUNCH_I4)
                DISPATCH_BT(32, LAUNCH_I4)
                default: TORCH_CHECK(false, "unsupported batch tile ", bt);
            }
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor int8_matmul(torch::Tensor x, torch::Tensor q, torch::Tensor scale) {
    TORCH_CHECK(x.is_cuda() && q.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(q.dim() == 2, "q must be [out, in]");
    TORCH_CHECK(q.scalar_type() == at::kChar, "q must be int8");

    auto x_c = x.contiguous();
    const int N = static_cast<int>(q.size(0));
    const int K = static_cast<int>(q.size(1));
    TORCH_CHECK(x_c.size(-1) == K, "x last dim must equal q.size(1)");
    TORCH_CHECK(K % 4 == 0, "in_features must be a multiple of 4");

    auto shape = x_c.sizes().vec();
    const int B = static_cast<int>(x_c.numel() / K);
    shape.back() = N;
    auto y = torch::empty(shape, x_c.options());
    if (B == 0 || N == 0) return y;

    auto x2 = x_c.reshape({B, K});
    auto q_c = q.contiguous();
    auto s_c = scale.contiguous();

    const int threads = 256;
    const int wpb = warps_per_block(threads);

    const at::cuda::OptionalCUDAGuard guard(device_of(x_c));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x_c.scalar_type(), "int8_matmul", [&] {
            int bt = 1;
            while (bt < B && bt < 32) bt <<= 1;
            const int tiles = (B + bt - 1) / bt;
            dim3 grid((N + wpb - 1) / wpb, tiles);

            switch (bt) {
                DISPATCH_BT(1, LAUNCH_I8)
                DISPATCH_BT(2, LAUNCH_I8)
                DISPATCH_BT(4, LAUNCH_I8)
                DISPATCH_BT(8, LAUNCH_I8)
                DISPATCH_BT(16, LAUNCH_I8)
                DISPATCH_BT(32, LAUNCH_I8)
                default: TORCH_CHECK(false, "unsupported batch tile ", bt);
            }
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
