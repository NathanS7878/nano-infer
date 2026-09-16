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


// ---------------------------------------------------------------------------
// KERNEL 4b — head-group fused decode attention.
//
// WHY THIS EXISTS. The kernel above assigns one block per (sequence, QUERY
// head). With GQA 14q/2kv that means 7 blocks independently read the same KV
// rows. The repo previously reported that path at "11.1% of peak", measured
// against COMPULSORY bytes (each KV element counted once), and concluded the
// kernel was leaving 89% of the card on the table.
//
// That conclusion was wrong, and the experiment that corrects it is
// bench/kernel_attention.py --sharing. Hold the block count and the per-block
// work exactly constant, and vary only how many query heads share a KV head:
//
//   kv heads | n_rep | distinct | issued  |     us | issued GB/s | % of peak
//         14 |     1 |  235 MB  | 235 MB  | 1132.3 |       207.4 |     46.3%
//          2 |     7 | 33.6 MB  | 235 MB  |  687.9 |       341.4 |     76.2%
//          1 |    14 | 16.8 MB  | 235 MB  |  673.8 |       348.6 |     77.8%
//
// Wall time flattens once n_rep >= 7, which says the 7x duplicate reads ARE
// served by cache rather than DRAM. But a cache hit still costs a load
// instruction, an L1/L2 transaction and issue slots, and against the bytes it
// actually ASKS FOR, the kernel sits at 76% of peak. It was never 11% of the
// card. It was near the ceiling of a load path carrying 7x more traffic than
// the algorithm requires.
//
// So the fix is not "go faster". It is "ask for less". This kernel assigns one
// block per (sequence, KV head) and lets the R = n_rep query heads sharing that
// KV head ride along: each K element is loaded once into a register and fed to
// R dot products, each V element loaded once and fed to R accumulators. Issued
// traffic falls by R; the arithmetic is unchanged, so intensity rises from
// 0.5 to R/2 FLOP per byte.
//
// It also attacks the second finding (Gotcha #17, latency-bound). The online
// softmax recurrence is sequential across tiles and each tile costs several
// __syncthreads(). Fusing does not reduce barriers per tile (7 here vs 6
// there), but each barrier now covers R times as much work, so barriers per
// unit of work fall roughly 6x.
//
// WHAT IT COSTS, stated up front: the block count falls by R. At batch 32 that
// is 448 -> 64 blocks on 46 SMs, still enough to fill the card. At batch 1 it
// is 14 -> 2 blocks, which cannot. So this is a heuristic path, not a
// replacement, and the dispatcher picks between the two on block count.
// Fixing batch 1 properly needs split-K, which is a separate change.
//
// The R query heads run in lockstep through one recurrence, so m, l and the
// correction factor become length-R register arrays. Every thread holds
// identical copies of them (they are outputs of block-wide reductions), which
// is why corr has to be staged through shared memory before anything indexes
// it by a thread-varying index: a dynamic index into a register array spills
// the array to local memory and undoes the point of the kernel.

// Block-wide reduction of R values at once. The barrier count is that of a
// SINGLE reduction: R warp-shuffle chains, then one shared round trip carrying
// R floats per warp. This is what stops the fusion from paying R times the
// synchronisation it exists to amortise.
template <bool IS_MAX, int R>
__inline__ __device__ void block_reduce_vec(float* v, float* scratch, float* bcast) {
    const int lane = threadIdx.x % WARP;
    const int warp = threadIdx.x / WARP;
    const int nwarps = (blockDim.x + WARP - 1) / WARP;

    #pragma unroll
    for (int r = 0; r < R; ++r)
        v[r] = IS_MAX ? warp_max(v[r]) : warp_sum(v[r]);

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < R; ++r) scratch[warp * R + r] = v[r];
    }
    __syncthreads();

    if (threadIdx.x < R) {
        float t = IS_MAX ? -INFINITY : 0.0f;
        for (int w = 0; w < nwarps; ++w) {
            const float s = scratch[w * R + threadIdx.x];
            t = IS_MAX ? fmaxf(t, s) : t + s;
        }
        bcast[threadIdx.x] = t;
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < R; ++r) v[r] = bcast[r];
}

template <typename scalar_t, int VEC, int R>
__global__ void decode_attention_grouped_kernel(
        const scalar_t* __restrict__ q,        // [batch, q_heads, head_dim]
        const scalar_t* __restrict__ k_pool,   // [slots, kv_heads, head_dim]
        const scalar_t* __restrict__ v_pool,   // [slots, kv_heads, head_dim]
        const int64_t*  __restrict__ slots,    // [batch, max_len]
        const int64_t*  __restrict__ lengths,  // [batch]
        scalar_t* __restrict__ out,            // [batch, q_heads, head_dim]
        const int q_heads, const int kv_heads, const int head_dim,
        const int max_len, const float scale, const bool vectorized) {

    extern __shared__ float smem[];
    const int TILE = blockDim.x;
    const int D = head_dim;

    float* q_s     = smem;                       // R*D
    float* acc     = q_s + R * D;                // R*D
    float* s_tile  = acc + R * D;                // R*TILE
    float* partial = s_tile + R * TILE;          // R*TILE, == ngrp*R*D
    float* scratch = partial + R * TILE;         // MAXW*R
    float* bcast   = scratch + MAXW * R;         // R
    float* corr_s  = bcast + R;                  // R
    int*   slot_s  = reinterpret_cast<int*>(corr_s + R);   // TILE

    const int b   = blockIdx.x / kv_heads;
    const int kvh = blockIdx.x % kv_heads;
    const int h0  = kvh * R;                     // first query head of the group
    const int tid = threadIdx.x;
    const int len = static_cast<int>(lengths[b]);

    for (int i = tid; i < R * D; i += TILE) {
        q_s[i] = static_cast<float>(
            q[(long long)(b * q_heads + h0 + i / D) * D + (i % D)]);
        acc[i] = 0.0f;
    }
    __syncthreads();

    if (len <= 0) {                              // nothing cached: output is 0
        for (int i = tid; i < R * D; i += TILE)
            out[(long long)(b * q_heads + h0 + i / D) * D + (i % D)] =
                static_cast<scalar_t>(0.0f);
        return;
    }

    const int d    = tid % D;                    // D % 32 == 0 is required, so
    const int grp  = tid / D;                    // grp is warp-uniform and the
    const int ngrp = TILE / D;                   // break below never diverges

    float m[R], l[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) { m[r] = -INFINITY; l[r] = 0.0f; }

    for (int base = 0; base < len; base += TILE) {
        const int pos = base + tid;

        // Stage the tile's slot indices once. Without this the PV phase below
        // re-reads slots[] from global on each of its ngrp-strided steps, and
        // every one of those is a dependent load standing in front of a V read.
        slot_s[tid] = (pos < len)
            ? static_cast<int>(slots[(long long)b * max_len + pos]) : -1;
        __syncthreads();

        // --- 1. R scores per position, from ONE pass over the K row ---------
        float s[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) s[r] = -INFINITY;

        if (pos < len) {
            const scalar_t* krow =
                k_pool + ((long long)slot_s[tid] * kv_heads + kvh) * D;
            float dot[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) dot[r] = 0.0f;

            // q_s is indexed by a loop counter, so every lane of a warp reads
            // the SAME address: a shared-memory broadcast, not R transactions.
            if (vectorized) {
                const float4* k4 = reinterpret_cast<const float4*>(krow);
                const int nvec = D / VEC;
                for (int c = 0; c < nvec; ++c) {
                    float4 chunk = k4[c];
                    const scalar_t* kv = reinterpret_cast<const scalar_t*>(&chunk);
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j) {
                        const float kval = static_cast<float>(kv[j]);
                        const int idx = c * VEC + j;
                        #pragma unroll
                        for (int r = 0; r < R; ++r)
                            dot[r] += q_s[r * D + idx] * kval;
                    }
                }
            } else {
                for (int idx = 0; idx < D; ++idx) {
                    const float kval = static_cast<float>(krow[idx]);
                    #pragma unroll
                    for (int r = 0; r < R; ++r)
                        dot[r] += q_s[r * D + idx] * kval;
                }
            }
            #pragma unroll
            for (int r = 0; r < R; ++r) s[r] = dot[r] * scale;
        }

        // --- 2. one block reduction carrying all R maxima -------------------
        float red[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) red[r] = s[r];
        block_reduce_vec<true, R>(red, scratch, bcast);

        float corr[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const float m_new = fmaxf(m[r], red[r]);
            corr[r] = expf(m[r] - m_new);
            m[r] = m_new;
            s[r] = (pos < len) ? expf(s[r] - m_new) : 0.0f;   // s now holds p
            s_tile[r * TILE + tid] = s[r];
            if (tid == r) corr_s[r] = corr[r];   // static index: stays in regs
        }

        block_reduce_vec<false, R>(s, scratch, bcast);        // s now holds sums
        #pragma unroll
        for (int r = 0; r < R; ++r) l[r] = l[r] * corr[r] + s[r];

        // --- 3. one pass over V feeding R accumulators ----------------------
        float local[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) local[r] = 0.0f;

        for (int j = grp; j < TILE; j += ngrp) {
            if (base + j >= len) break;
            const float vv = static_cast<float>(
                v_pool[((long long)slot_s[j] * kv_heads + kvh) * D + d]);
            #pragma unroll
            for (int r = 0; r < R; ++r) local[r] += s_tile[r * TILE + j] * vv;
        }
        #pragma unroll
        for (int r = 0; r < R; ++r)
            partial[(r * ngrp + grp) * D + d] = local[r];
        __syncthreads();

        // Rescale and accumulate in ONE pass: acc = acc*corr + sum. Splitting
        // that into a scale loop and an add loop would cost an extra barrier
        // per tile for nothing.
        for (int i = tid; i < R * D; i += TILE) {
            const int r = i / D, dd = i % D;
            float sum = 0.0f;
            for (int g = 0; g < ngrp; ++g) sum += partial[(r * ngrp + g) * D + dd];
            acc[i] = acc[i] * corr_s[r] + sum;
        }
        __syncthreads();
    }

    // --- 4. normalise once, at the very end --------------------------------
    // l has to go through shared for the same reason corr did: the divide is
    // indexed by i/D, which varies per thread, and a dynamic index into a
    // register array spills it. bcast is dead by now, so reuse it.
    #pragma unroll
    for (int r = 0; r < R; ++r) if (tid == r) bcast[r] = l[r];
    __syncthreads();

    for (int i = tid; i < R * D; i += TILE)
        out[(long long)(b * q_heads + h0 + i / D) * D + (i % D)] =
            static_cast<scalar_t>(acc[i] / bcast[i / D]);
}

// Shared-memory footprint of the grouped kernel, in bytes.
static inline size_t grouped_smem(int R, int D, int threads) {
    const size_t floats = 2 * (size_t)R * D      // q_s + acc
                        + 2 * (size_t)R * threads // s_tile + partial
                        + (size_t)MAXW * R        // warp scratch
                        + 2 * (size_t)R;          // bcast + corr_s
    return floats * sizeof(float) + (size_t)threads * sizeof(int);  // + slot_s
}

// Launch the grouped kernel for a compile-time R. Returns false if this (R, D,
// threads) does not fit in the device's opt-in shared memory, so the caller can
// fall back rather than launch something that will not run.
template <typename scalar_t, int VEC, int R>
static bool launch_grouped(const scalar_t* q, const scalar_t* k, const scalar_t* v,
                           const int64_t* slots, const int64_t* lengths,
                           scalar_t* out, int batch, int q_heads, int kv_heads,
                           int head_dim, int max_len, float scale, bool vectorized,
                           int threads, cudaStream_t stream) {
    auto* fn = decode_attention_grouped_kernel<scalar_t, VEC, R>;
    const int max_smem =
        at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;

    // The tile arrays scale with R*threads, so a large R can price a wide block
    // out of shared memory (R=16 at 1024 threads wants ~144 KB). Halve the block
    // until it fits rather than refusing outright -- a narrower grouped block
    // still beats the per-query-head kernel. Only give up below one group.
    while (threads > head_dim && grouped_smem(R, head_dim, threads) > (size_t)max_smem)
        threads /= 2;
    const size_t smem = grouped_smem(R, head_dim, threads);
    if (smem > (size_t)max_smem) return false;
    if (smem > 48u * 1024u) {
        // Anything over the 48 KB static limit has to be opted into explicitly.
        // sm_86 allows 99 KB per block; without this the launch fails outright.
        cudaError_t e = cudaFuncSetAttribute(
            fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) { cudaGetLastError(); return false; }
    }
    fn<<<batch * kv_heads, threads, smem, stream>>>(
        q, k, v, slots, lengths, out, q_heads, kv_heads, head_dim,
        max_len, scale, vectorized);
    return true;
}

// One switch case per instantiated n_rep. This has to be defined at file scope:
// AT_DISPATCH_FLOATING_TYPES_AND2 is itself a macro, so a #define written inside
// its body is consumed by that expansion rather than by the preprocessor.
#define NI_GROUPED_CASE(RVAL)                                                      case RVAL:                                                                         launched = launch_grouped<scalar_t, VEC, RVAL>(                                    q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(),                            v_c.data_ptr<scalar_t>(), slots_c.data_ptr<int64_t>(),                         len_c.data_ptr<int64_t>(), out.data_ptr<scalar_t>(),                           batch, q_heads, kv_heads, head_dim, max_len,                                   static_cast<float>(scale), vectorized, threads, stream);                   break;

torch::Tensor decode_attention_forward(
        torch::Tensor q, torch::Tensor k_pool, torch::Tensor v_pool,
        torch::Tensor slot_table, torch::Tensor lengths, double scale,
        int64_t block_size, int64_t fuse_heads) {

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

    auto out = torch::empty({batch, q_heads, head_dim}, q_c.options());
    if (batch == 0 || q_heads == 0) return out;

    const int n_rep = q_heads / kv_heads;
    const int sm_count =
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    // ---- which kernel -----------------------------------------------------
    // The grouped kernel reads each KV row once for all n_rep query heads
    // instead of n_rep times, but it launches n_rep times fewer blocks. That
    // trade is only worth taking while there are still enough blocks to fill
    // the SMs. Measured on an RTX 3070 (46 SMs), the crossover sits close to
    // one block per SM: at batch 32 the grouped path has 64 blocks and wins
    // large; at batch 1 it has 2 and loses badly. Requiring >= sm_count blocks
    // reproduces the measured choice on every case in the sweep.
    //
    // fuse_heads: 0 = auto, 1 = force grouped, -1 = force per-query-head. The
    // forcing modes exist so bench/kernel_attention.py can A/B the two paths on
    // identical inputs rather than trusting this heuristic.
    const int grouped_blocks = batch * kv_heads;
    const bool grouped_possible =
        n_rep > 1 && (head_dim % WARP == 0) && (grouped_blocks > 0);
    // Measured crossover (RTX 3070, 46 SMs, 14q/2kv, see the sweep in
    // PROGRESS.md): grouped loses below 8 blocks and wins above it, and the win
    // grows with block count -- 1.2x at 8 blocks, 2.5x at 32, 2.9x at 64. It is
    // block count that matters, not batch: what breaks the grouped path is
    // having fewer blocks than the machine can use, and it starts n_rep times
    // behind on that count.
    bool want_grouped =
        (fuse_heads > 0) ||
        (fuse_heads == 0 && grouped_possible && grouped_blocks >= 8);
    if (!grouped_possible) want_grouped = false;
    TORCH_CHECK(fuse_heads <= 0 || grouped_possible,
                "fuse_heads=1 requested but the grouped path does not apply "
                "(needs n_rep > 1 and head_dim a multiple of 32; got n_rep ",
                n_rep, ", head_dim ", head_dim, ")");

    // ---- block size -------------------------------------------------------
    // Per-query-head path: the sweep in PROGRESS.md showed this kernel is
    // LATENCY-bound, not bandwidth-bound. The online-softmax recurrence is
    // sequential across tiles and each tile costs several __syncthreads(), so
    // what hides that latency is warps per SM, which means a wider block when
    // there are few blocks to go around.
    //
    //   many blocks (batch 32 -> 448)  : 512 threads, measured best
    //   few blocks  (batch 1  ->  14)  : 1024 threads, 3.6x faster than 128
    //   short contexts                 : no point tiling wider than the sequence
    //
    // Grouped path: 512 measured best at batch 32; 1024 costs shared memory
    // (R*TILE floats twice over) without adding blocks.
    int auto_threads;
    if (want_grouped) {
        // The grouped optimum is a constant TOTAL thread count, not a constant
        // block size. Measured best block size across batch 4-64 x L 128-2048:
        //   8 blocks -> 1024,  16 -> 1024,  32 -> 1024,  64 -> 512,  128 -> 256
        // which is blocks * threads ~ 32768 in every row: about 1024 warps, or
        // 22 warps per SM on this card -- roughly half the 48-warp maximum.
        // Enough resident warps to hide the sequential recurrence, without
        // spending shared memory on tile arrays that cannot be filled. Dividing
        // a fixed budget reproduces the measured optimum, or comes within 9% of
        // it, at every point in that sweep.
        auto_threads = 1024;
        while (auto_threads > 128 && grouped_blocks * auto_threads > 32768)
            auto_threads /= 2;
    } else {
        auto_threads = (batch * q_heads >= 2 * sm_count) ? 512 : 1024;
        if (auto_threads > 256 && max_len <= 256) auto_threads = 256;
    }
    int threads = (block_size > 0) ? static_cast<int>(block_size) : auto_threads;
    TORCH_CHECK(threads > 0 && threads <= 1024 && threads % 32 == 0,
                "block_size must be a positive multiple of 32 up to 1024");
    // The accumulator update maps threads to (dim, group), so the block size
    // has to be a whole number of head_dim-wide groups.
    TORCH_CHECK(head_dim > 0 && threads % head_dim == 0 && head_dim <= threads,
                "head_dim must divide the block size ", threads,
                " (got head_dim ", head_dim, ")");

    const at::cuda::OptionalCUDAGuard guard(device_of(q_c));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q_c.scalar_type(), "decode_attention_forward", [&] {
            constexpr int VEC = 16 / sizeof(scalar_t);
            const bool vectorized =
                (head_dim % VEC == 0) &&
                ((static_cast<size_t>(head_dim) * sizeof(scalar_t)) % 16 == 0);

            bool launched = false;
            if (want_grouped) {
                // R is a template parameter so the R-wide register arrays stay
                // in registers. Only the ratios that occur in practice are
                // instantiated; anything else falls through to the general
                // per-query-head kernel, which handles every n_rep.
                switch (n_rep) {
                    // 6 is Qwen2.5-1.5B (12 query / 2 KV heads); 7 is 0.5B.
                    // A ratio missing here does not fail -- it silently runs
                    // the slower per-query-head kernel -- so add new models'
                    // ratios deliberately and test them.
                    NI_GROUPED_CASE(2)  NI_GROUPED_CASE(4)
                    NI_GROUPED_CASE(6)  NI_GROUPED_CASE(7)
                    NI_GROUPED_CASE(8)  NI_GROUPED_CASE(14)
                    NI_GROUPED_CASE(16)
                    default: launched = false;
                }
                TORCH_CHECK(launched || fuse_heads <= 0,
                            "fuse_heads=1 requested but n_rep ", n_rep,
                            " has no grouped instantiation, or the block would "
                            "need more shared memory than this device allows");
            }

            if (!launched) {
                // q_s + acc + s_tile + partial + warp scratch
                const size_t smem =
                    (2 * head_dim + 2 * threads + MAXW) * sizeof(float);
                decode_attention_kernel<scalar_t, VEC>
                    <<<batch * q_heads, threads, smem, stream>>>(
                        q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(),
                        v_c.data_ptr<scalar_t>(), slots_c.data_ptr<int64_t>(),
                        len_c.data_ptr<int64_t>(), out.data_ptr<scalar_t>(),
                        q_heads, kv_heads, head_dim, max_len, n_rep,
                        static_cast<float>(scale), vectorized);
            }
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

#undef NI_GROUPED_CASE
