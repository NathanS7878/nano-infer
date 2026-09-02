// Fused RoPE — nano-infer Phase 3, kernel 3.
//
// Rotary position embedding rotates each query/key vector by its position:
//
//     x_rot = x * cos + rotate_half(x) * sin
//     rotate_half([x1, x2]) = [-x2, x1]        (x1 = low half, x2 = high half)
//
// PyTorch runs that as FIVE kernels, and one of them is a `cat` that
// materializes a whole extra tensor. Counting bytes per element of x (E
// elements, 2 bytes each; cos/sin are small enough to live in L2):
//
//     -x2                read E/2, write E/2                    =  2E bytes
//     cat((-x2, x1))     read E,   write E                      =  4E
//     x * cos            read E,   write E                      =  4E
//     rotated * sin      read E,   write E                      =  4E
//     t1 + t2            read 2E,  write E                      =  6E
//     ----------------------------------------------------------------------
//     total                                                       20E bytes
//
// Ours reads x once and writes once: 4E bytes. Predicted ceiling 20/4 = 5x —
// bigger than SwiGLU's 1.67x, and for a knowable reason: rotate_half is pure
// data movement. It computes nothing. It exists only to get the operand into a
// layout the elementwise multiply can consume, and a fused kernel does that
// with an index expression instead of a memory round trip.
//
// THE PAIRING IS THE PLACE THIS GOES WRONG. HF/Llama pairs dim i with
// i + head_dim/2. The other plausible convention — pairing adjacent dims
// (0,1), (2,3), ... — is what the original RoPE paper draws, and it is also
// what the cos/sin tables here are NOT built for: build_rope_cache duplicates
// the frequencies as cat(freqs, freqs) precisely to match the halves layout.
// Getting it backwards produces a model that still runs, still emits fluent
// text, and is quietly wrong about position. There is no exception to catch,
// which is why the parity test below checks the pairing directly rather than
// trusting an end-to-end result to notice.
//
// STRUCTURE: one thread owns a rotation PAIR (j, j + half), so it loads x1 and
// x2 once and produces both outputs. No element is read twice, and the
// rotation costs an index offset rather than a tensor.
//
// NUMERICS: the reference rounds to fp16 after EACH multiply, then adds.
// Carrying full fp32 through the add would be more accurate and therefore
// wrong — it would not match. Match, do not improve. (The fp32 products are
// themselves exact: two fp16 significands multiply to 22 bits and fp32 holds
// 24, so rounding once to fp16 is identical to a true fp16 multiply.)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// out = fp16(fp16(a * c) + fp16(b * s)) — mirrors x*cos + rotate_half(x)*sin
// with the reference's rounding points.
template <typename scalar_t>
__inline__ __device__ scalar_t rope_blend(float a, float c, float b, float s) {
    const scalar_t t1 = static_cast<scalar_t>(a * c);
    const scalar_t t2 = static_cast<scalar_t>(b * s);
    return static_cast<scalar_t>(static_cast<float>(t1) + static_cast<float>(t2));
}

// Decompose a flat row index over [batch, heads, n] into the cos/sin row.
// cos/sin are either [n, head_dim] (one shared position range, Phase 1) or
// [batch, n, head_dim] (per-sequence positions, required by Phase 2 step 3's
// continuous batching, where one sequence sits at position 60 while its
// neighbour is at position 3).
__inline__ __device__ long long cos_row_for(
        long long row, int heads, int n, bool per_batch) {
    const long long n_idx = row % n;
    if (!per_batch) return n_idx;
    const long long b_idx = (row / n) / heads;
    return b_idx * n + n_idx;
}

template <typename scalar_t>
__global__ void rope_kernel(
        const scalar_t* __restrict__ x,
        const scalar_t* __restrict__ cosb,
        const scalar_t* __restrict__ sinb,
        scalar_t* __restrict__ out,
        const long long total_units,
        const int heads, const int n, const int head_dim, const int half,
        const bool per_batch) {

    for (long long u = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         u < total_units; u += (long long)gridDim.x * blockDim.x) {

        const long long row = u / half;
        const int j = static_cast<int>(u % half);

        const scalar_t* __restrict__ xr = x + row * head_dim;
        scalar_t* __restrict__ orow = out + row * head_dim;
        const long long crow = cos_row_for(row, heads, n, per_batch) * head_dim;

        const float x1 = static_cast<float>(xr[j]);
        const float x2 = static_cast<float>(xr[j + half]);

        // low half gets -x2 as its rotate_half partner, high half gets +x1
        orow[j] = rope_blend<scalar_t>(
            x1, static_cast<float>(cosb[crow + j]),
            -x2, static_cast<float>(sinb[crow + j]));
        orow[j + half] = rope_blend<scalar_t>(
            x2, static_cast<float>(cosb[crow + j + half]),
            x1, static_cast<float>(sinb[crow + j + half]));
    }
}

// ---------------------------------------------------------------------------
// Vectorized variant — VEC halves per thread per half-row. Same lesson as
// kernels 1 and 2: 2 bytes in flight per thread cannot saturate the memory
// system. head_dim 64 -> half 32 -> 4 threads cover a row with float4 loads.
//
// Alignment holds by construction: a row is head_dim*2 = 128 bytes, the low
// chunk sits at 16*chunk bytes and the high chunk at 64 + 16*chunk.
// ---------------------------------------------------------------------------

template <typename scalar_t, int VEC>
__global__ void rope_vec_kernel(
        const scalar_t* __restrict__ x,
        const scalar_t* __restrict__ cosb,
        const scalar_t* __restrict__ sinb,
        scalar_t* __restrict__ out,
        const long long total_units,
        const int heads, const int n, const int head_dim, const int half,
        const int chunks_per_row, const bool per_batch) {

    using Vec = float4;

    for (long long u = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         u < total_units; u += (long long)gridDim.x * blockDim.x) {

        const long long row = u / chunks_per_row;
        const int chunk = static_cast<int>(u % chunks_per_row);
        const int lo = chunk * VEC;
        const int hi = half + lo;

        const scalar_t* xr = x + row * head_dim;
        scalar_t* orow = out + row * head_dim;
        const long long crow = cos_row_for(row, heads, n, per_batch) * head_dim;
        const scalar_t* cr = cosb + crow;
        const scalar_t* sr = sinb + crow;

        Vec xlo = *reinterpret_cast<const Vec*>(xr + lo);
        Vec xhi = *reinterpret_cast<const Vec*>(xr + hi);
        Vec clo = *reinterpret_cast<const Vec*>(cr + lo);
        Vec chi = *reinterpret_cast<const Vec*>(cr + hi);
        Vec slo = *reinterpret_cast<const Vec*>(sr + lo);
        Vec shi = *reinterpret_cast<const Vec*>(sr + hi);

        const scalar_t* xv1 = reinterpret_cast<const scalar_t*>(&xlo);
        const scalar_t* xv2 = reinterpret_cast<const scalar_t*>(&xhi);
        const scalar_t* cv1 = reinterpret_cast<const scalar_t*>(&clo);
        const scalar_t* cv2 = reinterpret_cast<const scalar_t*>(&chi);
        const scalar_t* sv1 = reinterpret_cast<const scalar_t*>(&slo);
        const scalar_t* sv2 = reinterpret_cast<const scalar_t*>(&shi);

        Vec olo, ohi;
        scalar_t* ov1 = reinterpret_cast<scalar_t*>(&olo);
        scalar_t* ov2 = reinterpret_cast<scalar_t*>(&ohi);

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float x1 = static_cast<float>(xv1[j]);
            const float x2 = static_cast<float>(xv2[j]);
            ov1[j] = rope_blend<scalar_t>(x1, static_cast<float>(cv1[j]),
                                          -x2, static_cast<float>(sv1[j]));
            ov2[j] = rope_blend<scalar_t>(x2, static_cast<float>(cv2[j]),
                                          x1, static_cast<float>(sv2[j]));
        }

        *reinterpret_cast<Vec*>(orow + lo) = olo;
        *reinterpret_cast<Vec*>(orow + hi) = ohi;
    }
}

static inline bool rope_is_aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

torch::Tensor rope_forward(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
    TORCH_CHECK(x.is_cuda() && cos.is_cuda() && sin.is_cuda(),
                "x, cos and sin must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be [batch, heads, n, head_dim]");
    TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin must have the same shape");
    TORCH_CHECK(cos.dim() == 2 || cos.dim() == 3,
                "cos/sin must be [n, head_dim] or [batch, n, head_dim]");
    TORCH_CHECK(x.scalar_type() == cos.scalar_type() &&
                x.scalar_type() == sin.scalar_type(), "dtype mismatch");

    auto x_c = x.contiguous();
    auto cos_c = cos.contiguous();
    auto sin_c = sin.contiguous();

    const int batch = static_cast<int>(x_c.size(0));
    const int heads = static_cast<int>(x_c.size(1));
    const int n = static_cast<int>(x_c.size(2));
    const int head_dim = static_cast<int>(x_c.size(3));
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
    const int half = head_dim / 2;

    const bool per_batch = (cos_c.dim() == 3);
    TORCH_CHECK(cos_c.size(-1) == head_dim, "cos/sin last dim must equal head_dim");
    TORCH_CHECK(cos_c.size(-2) == n, "cos/sin position dim must equal x.size(2)");
    if (per_batch) {
        TORCH_CHECK(cos_c.size(0) == batch, "cos/sin batch dim must equal x.size(0)");
    }

    auto out = torch::empty_like(x_c);
    const long long rows = (long long)batch * heads * n;
    if (rows == 0 || head_dim == 0) return out;

    const int threads = 256;
    const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int max_blocks = sm_count * 32;

    const at::cuda::OptionalCUDAGuard guard(device_of(x_c));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x_c.scalar_type(), "rope_forward", [&] {
            const scalar_t* xp = x_c.data_ptr<scalar_t>();
            const scalar_t* cp = cos_c.data_ptr<scalar_t>();
            const scalar_t* sp = sin_c.data_ptr<scalar_t>();
            scalar_t* op = out.data_ptr<scalar_t>();

            constexpr int VEC = 16 / sizeof(scalar_t);
            const bool vectorizable =
                (half % VEC == 0) &&
                ((static_cast<size_t>(head_dim) * sizeof(scalar_t)) % 16 == 0) &&
                rope_is_aligned16(xp) && rope_is_aligned16(cp) &&
                rope_is_aligned16(sp) && rope_is_aligned16(op);

            if (vectorizable) {
                const int chunks_per_row = half / VEC;
                const long long units = rows * chunks_per_row;
                long long want = (units + threads - 1) / threads;
                const int blocks = static_cast<int>(want < max_blocks ? want : max_blocks);
                rope_vec_kernel<scalar_t, VEC><<<blocks, threads, 0, stream>>>(
                    xp, cp, sp, op, units, heads, n, head_dim, half,
                    chunks_per_row, per_batch);
            } else {
                const long long units = rows * half;
                long long want = (units + threads - 1) / threads;
                const int blocks = static_cast<int>(want < max_blocks ? want : max_blocks);
                rope_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    xp, cp, sp, op, units, heads, n, head_dim, half, per_batch);
            }
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
