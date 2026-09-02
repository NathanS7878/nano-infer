// Fused decode attention with online softmax — nano-infer Phase 3, kernel 4.
//
// This kernel is different IN KIND from kernels 1-3. Those were fusions: the
// win was byte-counted in advance and the reference was a few elementwise ops
// glued together. This one changes the ALGORITHM. The reference materializes a
// [batch, heads, 1, L] score matrix, softmaxes it, then multiplies by V. Online
// softmax never materializes it at all.
//
// THE RECURRENCE (verified in Python before a line of CUDA was written).
// Streaming keys in tiles, per (batch, query head):
//
//     m   running max of scores seen so far      (init -inf)
//     l   running sum of exp(s - m)              (init 0)
//     acc running sum of exp(s - m) * v          (init 0, head_dim wide)
//
//     per tile:  m_new = max(m, max(s_tile))
//                corr  = exp(m - m_new)
//                l     = l * corr + sum(exp(s_tile - m_new))
//                acc   = acc * corr + sum_j exp(s_j - m_new) * v_j
//                m     = m_new
//     out = acc / l
//
// `corr` is the whole trick. A softmax needs the max of the entire row for
// stability, but a streaming kernel does not have it yet. So it uses the max so
// far, and when a later tile raises it, everything already accumulated is
// retroactively rescaled by exp(m_old - m_new). Dropping `corr` still produces
// finite, plausible output — the Python prototype measured 2.881 absolute error
// from that bug alone, and there is a regression test for it.
//
// WHY THIS IS THE KERNEL THAT MATTERS. Phase 2 measured decode achieving 26.5
// GB/s = 5.9% of this card's 448 GB/s peak. Kernels 1-3 could not move that:
// at decode they operate on tensors of a few tens of KB, so they are
// launch-bound, not bandwidth-bound. This kernel streams the entire KV cache,
// which IS the decode bottleneck.
//
// BYTE COUNT, per sequence per layer, L cached positions, fp16, GQA 14q/2kv:
//
//   ours:  read K   L * 2 kv_heads * 64 * 2 B  = 256L
//          read V   same                        = 256L
//          q, out   ~1.8 KB each, negligible
//          ---------------------------------------------
//          total                                  512L
//
//   the PyTorch paged reference:
//          cache.gather   read 512L, write 512L         = 1024L
//          repeat_kv      read 512L, write 3584L        = 4096L   <-- dominant
//          q @ k^T        read 3584L, write 28L         = 3612L
//          float/mask/softmax/cast round trips          =  392L
//          probs @ v      read 3584L + 28L              = 3612L
//          ---------------------------------------------------
//          total                                        ~12736L
//
// Predicted ceiling ~24x. MEASURED: 2.48x. That miss is the most useful result
// in Phase 3 and is not swept under the rug -- see the scaling experiment in
// PROGRESS.md. Byte counting predicts a ceiling only for a kernel that is
// actually bandwidth-bound, and this one is LATENCY-bound: holding the KV pool
// fixed and varying only how many query heads share it, 2 -> 8 query heads
// quadruples the blocks and leaves wall time flat (281.8 -> 292.0 us). The
// online-softmax recurrence is sequential across tiles and each tile costs
// several __syncthreads(), so what hides the latency is warps per SM. Tuning
// the block size accordingly (see the auto heuristic below) took this from
// 7.9% to 11.1% of peak, and 3.6x at batch 1.
//
// Still to do, and recorded rather than claimed: split-K (partition L across
// blocks, combine partial (m, l, acc) -- what real flash-decoding does for long
// context with few sequences), and one block per KV head instead of per query
// head so the 7 heads sharing a KV head read it once between them.
//
// The single largest term in the reference is `repeat_kv`, which
// materializes a 7x copy of K and V so that a batched matmul can see 14 KV
// heads that GQA deliberately did not store. That is kernel 3's lesson again,
// louder: the most expensive thing in the reference is not the math, it is the
// plumbing that reshapes operands to suit a library call. This kernel reads KV
// head h/7 directly from an index expression and copies nothing.
//
// ARITHMETIC INTENSITY: 3584L FLOPs over 512L bytes = 7 FLOP/byte, against this
// card's ~364 FLOP/byte roofline ridge. Memory-bound by a factor of ~52 — so
// still memory-bound, but note GQA RAISED the intensity: with 14 KV heads
// instead of 2 the same math would move 3584L bytes for 1 FLOP/byte. GQA's
// purpose is exactly this, and it is visible in one ratio.
//
// PAGED IN PLACE: the kernel walks the slot table directly instead of gathering
// a contiguous KV view first, removing the copy that Phase 2's paged cache pays
// every step. That is a second, independent win; bench/kernel_attention.py
// measures it separately so the two do not blur into one number.
//
// NUMERICS: this is the first kernel in the project that CANNOT be bit-identical
// to its reference, and the reason is structural, not sloppy. The reference
// rounds the QK product to fp16, softmaxes in fp32, rounds the probabilities
// back to fp16, then accumulates. An online algorithm cannot round the
// probabilities the same way, because it does not know the normaliser until the
// end. So parity is argued differently here: we compute an fp64 ground truth and
// show our error against it is NO LARGER than the reference's (measured:
// 0.48-1.00x on every shape). That bar cannot be satisfied by being wrong in the
// same direction as PyTorch, which "close to the reference" would allow. See
// tests/test_kernels.py::test_decode_attention_correctness.
//
// The fp64 ground truth is computed ON THE CPU, deliberately: torch.softmax in
// float64 on CUDA returns wrong results on this machine for any multi-row
// tensor (rows summing to 0.68 instead of 1.0). That bug made this kernel look
// wrong by 0.18 until the oracle itself was tested. See
// tests/test_kernels.py::test_fp64_softmax_on_cuda_is_unreliable.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define WARP 32
#define MAXW 32

__inline__ __device__ float warp_max(float v) {
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}

__inline__ __device__ float warp_sum(float v) {
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Block-wide reduction through one shared slot per warp, then warp 0 combines.
// `scratch` must hold at least num_warps floats; result is broadcast to all.
template <bool IS_MAX>
__inline__ __device__ float block_reduce(float v, float* scratch, float* bcast) {
    const int lane = threadIdx.x % WARP;
    const int warp = threadIdx.x / WARP;
    const int nwarps = (blockDim.x + WARP - 1) / WARP;

    v = IS_MAX ? warp_max(v) : warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();

    if (warp == 0) {
        float t = (lane < nwarps) ? scratch[lane]
                                  : (IS_MAX ? -INFINITY : 0.0f);
        t = IS_MAX ? warp_max(t) : warp_sum(t);
        if (lane == 0) *bcast = t;
    }
    __syncthreads();
    return *bcast;
}

template <typename scalar_t, int VEC>
__global__ void decode_attention_kernel(
        const scalar_t* __restrict__ q,        // [batch, q_heads, head_dim]
        const scalar_t* __restrict__ k_pool,   // [slots, kv_heads, head_dim]
        const scalar_t* __restrict__ v_pool,   // [slots, kv_heads, head_dim]
        const int64_t*  __restrict__ slots,    // [batch, max_len]
        const int64_t*  __restrict__ lengths,  // [batch]
        scalar_t* __restrict__ out,            // [batch, q_heads, head_dim]
        const int q_heads, const int kv_heads, const int head_dim,
        const int max_len, const int n_rep, const float scale,
        const bool vectorized) {

    extern __shared__ float smem[];
    float* q_s     = smem;                       // head_dim
    float* acc     = q_s + head_dim;             // head_dim
    float* s_tile  = acc + head_dim;             // blockDim.x
    float* partial = s_tile + blockDim.x;        // head_dim * groups
    float* scratch = partial + blockDim.x;       // >= num_warps (blockDim.x >= head_dim*groups)
    __shared__ float bcast;

    const int bh = blockIdx.x;
    const int b = bh / q_heads;
    const int h = bh % q_heads;
    const int kvh = h / n_rep;                   // GQA: no copy, just an index
    const int tid = threadIdx.x;
    const int TILE = blockDim.x;

    const int len = static_cast<int>(lengths[b]);

    for (int i = tid; i < head_dim; i += TILE) {
        q_s[i] = static_cast<float>(q[(long long)(b * q_heads + h) * head_dim + i]);
        acc[i] = 0.0f;
    }
    __syncthreads();

    if (len <= 0) {                              // nothing cached: define output as 0
        for (int i = tid; i < head_dim; i += TILE)
            out[(long long)(b * q_heads + h) * head_dim + i] = static_cast<scalar_t>(0.0f);
        return;
    }

    // thread -> (dim, group) mapping for the accumulator update
    const int d = tid % head_dim;
    const int grp = tid / head_dim;
    const int ngrp = TILE / head_dim;

    float m = -INFINITY;
    float l = 0.0f;

    for (int base = 0; base < len; base += TILE) {
        const int pos = base + tid;

        // --- 1. score for this thread's position: s = (q . k_pos) * scale ---
        float s = -INFINITY;
        if (pos < len) {
            const long long slot = slots[(long long)b * max_len + pos];
            const scalar_t* krow =
                k_pool + (slot * kv_heads + kvh) * (long long)head_dim;
            float dot = 0.0f;
            if (vectorized) {
                const float4* k4 = reinterpret_cast<const float4*>(krow);
                const int nvec = head_dim / VEC;
                for (int c = 0; c < nvec; ++c) {
                    float4 chunk = k4[c];
                    const scalar_t* kv = reinterpret_cast<const scalar_t*>(&chunk);
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j)
                        dot += q_s[c * VEC + j] * static_cast<float>(kv[j]);
                }
            } else {
                for (int j = 0; j < head_dim; ++j)
                    dot += q_s[j] * static_cast<float>(krow[j]);
            }
            s = dot * scale;
        }

        // --- 2. tile max, then the online update of m and the correction ---
        const float tile_max = block_reduce<true>(s, scratch, &bcast);
        const float m_new = fmaxf(m, tile_max);
        // m = -inf on the first tile gives exp(-inf) = 0, which is correct:
        // there is nothing accumulated yet to rescale. m_new is always finite
        // because the loop only runs while base < len, so every tile holds at
        // least one unmasked position.
        const float corr = expf(m - m_new);

        const float p = (pos < len) ? expf(s - m_new) : 0.0f;
        s_tile[tid] = p;
        const float tile_sum = block_reduce<false>(p, scratch, &bcast);

        l = l * corr + tile_sum;
        m = m_new;

        // --- 3. acc = acc * corr + sum_j p_j * v_j ---
        for (int i = tid; i < head_dim; i += TILE) acc[i] *= corr;
        __syncthreads();

        float local = 0.0f;
        for (int j = grp; j < TILE; j += ngrp) {
            const int pj = base + j;
            if (pj >= len) break;                // uniform within a warp: grp is
            const float pv = s_tile[j];          // constant across a warp's lanes
            if (pv != 0.0f) {
                const long long slot = slots[(long long)b * max_len + pj];
                local += pv * static_cast<float>(
                    v_pool[(slot * kv_heads + kvh) * (long long)head_dim + d]);
            }
        }
        partial[grp * head_dim + d] = local;
        __syncthreads();

        if (tid < head_dim) {
            float sum = 0.0f;
            for (int g = 0; g < ngrp; ++g) sum += partial[g * head_dim + tid];
            acc[tid] += sum;
        }
        __syncthreads();
    }

    // --- 4. normalise once, at the very end ---
    for (int i = tid; i < head_dim; i += TILE)
        out[(long long)(b * q_heads + h) * head_dim + i] =
            static_cast<scalar_t>(acc[i] / l);
}

torch::Tensor decode_attention_forward(
        torch::Tensor q, torch::Tensor k_pool, torch::Tensor v_pool,
        torch::Tensor slot_table, torch::Tensor lengths, double scale,
        int64_t block_size) {

    TORCH_CHECK(q.is_cuda() && k_pool.is_cuda() && v_pool.is_cuda(),
                "q, k_pool and v_pool must be CUDA tensors");
    TORCH_CHECK(q.dim() == 3, "q must be [batch, q_heads, head_dim]");
    TORCH_CHECK(k_pool.dim() == 3 && v_pool.dim() == 3,
                "k_pool/v_pool must be [slots, kv_heads, head_dim]");
    TORCH_CHECK(slot_table.dim() == 2, "slot_table must be [batch, max_len]");
    TORCH_CHECK(lengths.dim() == 1, "lengths must be [batch]");
    TORCH_CHECK(q.scalar_type() == k_pool.scalar_type() &&
                q.scalar_type() == v_pool.scalar_type(), "dtype mismatch");

    auto q_c = q.contiguous();
    auto k_c = k_pool.contiguous();
    auto v_c = v_pool.contiguous();
    auto slots_c = slot_table.to(torch::kLong).contiguous();
    auto len_c = lengths.to(torch::kLong).contiguous();

    const int batch = static_cast<int>(q_c.size(0));
    const int q_heads = static_cast<int>(q_c.size(1));
    const int head_dim = static_cast<int>(q_c.size(2));
    const int kv_heads = static_cast<int>(k_c.size(1));
    const int max_len = static_cast<int>(slots_c.size(1));

    TORCH_CHECK(k_c.size(2) == head_dim && v_c.size(2) == head_dim,
                "head_dim mismatch between q and the KV pool");
    TORCH_CHECK(v_c.size(1) == kv_heads, "kv_heads mismatch between K and V pool");
    TORCH_CHECK(q_heads % kv_heads == 0, "q_heads must be a multiple of kv_heads");
    TORCH_CHECK(slots_c.size(0) == batch && len_c.size(0) == batch,
                "slot_table/lengths batch mismatch");

    // The accumulator update maps threads to (dim, group), so the block size has
    // to be a whole number of head_dim-wide groups. It is also the tile width,
    // which is why it is tunable: see the block-size sweep in PROGRESS.md.
    // Auto block size. The sweep in PROGRESS.md showed this kernel is
    // LATENCY-bound, not bandwidth-bound: holding KV volume fixed and
    // quadrupling the block count left wall time flat. The online-softmax
    // recurrence is sequential across tiles and each tile costs several
    // __syncthreads(), so what hides that latency is warps per SM — which means
    // a wider block when there are few blocks to go around.
    //
    //   many blocks (batch 32 -> 448)  : 512 threads, measured best
    //   few blocks  (batch 1  ->  14)  : 1024 threads, 3.6x faster than 128
    //   short contexts                 : no point tiling wider than the sequence
    //
    // This reproduces the measured optimum on every case in that sweep.
    const int sm_count_for_bs =
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int auto_threads = (batch * q_heads >= 2 * sm_count_for_bs) ? 512 : 1024;
    if (auto_threads > 256 && max_len <= 256) auto_threads = 256;
    const int threads = (block_size > 0) ? static_cast<int>(block_size) : auto_threads;
    TORCH_CHECK(threads > 0 && threads <= 1024 && threads % 32 == 0,
                "block_size must be a positive multiple of 32 up to 1024");
    TORCH_CHECK(head_dim > 0 && threads % head_dim == 0 && head_dim <= threads,
                "head_dim must divide the block size ", threads,
                " (got head_dim ", head_dim, ")");

    auto out = torch::empty({batch, q_heads, head_dim}, q_c.options());
    if (batch == 0 || q_heads == 0) return out;

    const int n_rep = q_heads / kv_heads;
    const at::cuda::OptionalCUDAGuard guard(device_of(q_c));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q_c.scalar_type(), "decode_attention_forward", [&] {
            constexpr int VEC = 16 / sizeof(scalar_t);
            const bool vectorized =
                (head_dim % VEC == 0) &&
                ((static_cast<size_t>(head_dim) * sizeof(scalar_t)) % 16 == 0);

            // q_s + acc + s_tile + partial + warp scratch
            const size_t smem = (2 * head_dim + 2 * threads + MAXW) * sizeof(float);

            decode_attention_kernel<scalar_t, VEC>
                <<<batch * q_heads, threads, smem, stream>>>(
                    q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(), slots_c.data_ptr<int64_t>(),
                    len_c.data_ptr<int64_t>(), out.data_ptr<scalar_t>(),
                    q_heads, kv_heads, head_dim, max_len, n_rep,
                    static_cast<float>(scale), vectorized);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
