// EXPERIMENTAL, NOT PART OF THE ENGINE BUILD -- a documented dead end.
//
// Built only by bench/hmma_probe.py, as its own extension. Kept so the negative
// result recorded in ROADMAP #40 is reproducible (hard rule 3), not so it can be
// used.
//
// THE GOAL was a tensor-core quantized matmul: ROADMAP #38 showed INT4 decode at
// 0.16x fp16 at batch 32 under CUDA graphs, with the loss matching the ~8x gap
// between this card's scalar fp32 (~20.3 TFLOP/s) and fp16 tensor-core
// (~163 TFLOP/s) throughput. The plan was to dequantize one weight tile at a
// time into shared memory and multiply it on tensor cores via nvcuda::wmma
// (<mma.h>): fragments plus mma_sync(D, A, B, C) = A.B + C.
//
// WHAT HAPPENED. Step 1 -- a plain fp16 X.W^T with no quantization, to pin the
// undocumented fragment layout against torch -- returned finite, plausibly
// sized outputs 20-60% off on every shape. The probe below then showed the
// problem is not layout:
//
//   * all 16 interpretations (A row/col x B row/col x store row/col x aliased
//     or separate C) match NO candidate product (X.W^T, its transpose, X.W,
//     W.X, ...), on small-integer inputs whose products are fp16-exact;
//   * setting any single input element to 1 lights up the SAME output cells
//     regardless of which element it was;
//   * all-zero inputs produce nonzero output, and with an aliased accumulator
//     not even the same output twice;
//   * y(2X) != 2 y(X): the op is not bilinear.
//
// And the definitions do not describe dense matrices: the m16n16k16 fp16
// matrix fragment derives from __frag_base<__half, 16> -- 16 storage elements,
// not 256 -- and its accumulator holds 8. The header (crt/mma.h[pp]) is marked
// proprietary and "internal ... must not be used directly", and documents no
// element semantics. The fragments' real meaning could not be recovered from it,
// so this path was abandoned rather than reverse-engineered by probing bit
// patterns: nothing learned that way would be a defensible claim.
//
// Nathan's prediction for the kernel (compute-bound; beats graphed fp16 at
// batch 32) was therefore NOT TESTED -- neither confirmed nor refuted.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

#define TILE 16

__global__ void hmma_matmul_f16_kernel(
        const __half* __restrict__ x,   // [B, K] fp16, row-major
        const __half* __restrict__ w,   // [N, K] fp16, row-major
        __half* __restrict__ y,         // [B, N] fp16, row-major
        const int B, const int N, const int K) {

    const int b0 = blockIdx.x * TILE;
    const int n0 = blockIdx.y * TILE;
    if (b0 >= B || n0 >= N) return;

    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> d;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bm;
    // torch's extension build defines __CUDA_NO_HALF_CONVERSIONS__, which compiles
    // out every numeric __half constructor. fp16 +0.0 is all-zero bits, so build
    // it from the raw bit pattern instead of converting.
    __half_raw zero_bits;
    zero_bits.x = 0;
    wmma::fill_fragment(d, __half(zero_bits));

    for (int k0 = 0; k0 < K; k0 += TILE) {
        wmma::load_matrix_sync(a, x + (size_t)b0 * K + k0, K);
        wmma::load_matrix_sync(bm, w + (size_t)n0 * K + k0, K);
        wmma::mma_sync(d, a, bm, d);
    }
    wmma::store_matrix_sync(y + (size_t)b0 * N + n0, d, N, wmma::mem_row_major);
}

torch::Tensor hmma_matmul_f16(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16 && w.scalar_type() == torch::kFloat16,
                "step 1 is fp16 only");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "x [B, K], w [N, K]");
    auto xc = x.contiguous();
    auto wc = w.contiguous();
    const int B = xc.size(0), K = xc.size(1), N = wc.size(0);
    TORCH_CHECK(wc.size(1) == K, "x and w disagree on in_features");
    // step 1 proves layout only; padding for ragged shapes is a later step
    TORCH_CHECK(B % TILE == 0 && N % TILE == 0 && K % TILE == 0,
                "step 1 requires every dimension to be a multiple of 16, got B=",
                B, " N=", N, " K=", K);

    auto y = torch::empty({B, N}, xc.options());
    const at::cuda::OptionalCUDAGuard guard(device_of(xc));
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 blocks((unsigned)(B / TILE), (unsigned)(N / TILE));
    hmma_matmul_f16_kernel<<<blocks, 1, 0, stream>>>(
        reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(wc.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()), B, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// ---------------------------------------------------------------------------
// hmma_layout_probe -- run ONE 16x16x16 tile under every layout interpretation.
//
// Step 1 as first written was wrong: its output was finite and of the right
// magnitude but 20-60% off, the signature of a layout error. With B = N = K = 16
// the leading dimension is 16 under any reading, so only major-ness is in
// question: A row/col (2) x B row/col (2) x store row/col (2) = 8 layouts, plus
// whether mma_sync tolerates D aliasing its own C input. `mode` bits:
//   bit 0: A col_major     bit 1: B col_major     bit 2: store mem_col_major
//   bit 3: use a SEPARATE C fragment (no aliasing)
// The Python side compares every mode against candidate products. This op is
// kept, with a test, because it documents an interface NVIDIA does not.
// ---------------------------------------------------------------------------

template <typename LA, typename LB>
__global__ void hmma_probe_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                                  __half* __restrict__ y, const bool store_col,
                                  const bool separate_c) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> d;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, LA> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, LB> bm;
    __half_raw zero_bits;
    zero_bits.x = 0;
    wmma::fill_fragment(d, __half(zero_bits));
    wmma::fill_fragment(c, __half(zero_bits));
    wmma::load_matrix_sync(a, x, 16);
    wmma::load_matrix_sync(bm, w, 16);
    if (separate_c) wmma::mma_sync(d, a, bm, c);
    else            wmma::mma_sync(d, a, bm, d);
    wmma::store_matrix_sync(y, d, 16, store_col ? wmma::mem_col_major : wmma::mem_row_major);
}

torch::Tensor hmma_layout_probe(torch::Tensor x, torch::Tensor w, int64_t mode) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.scalar_type() == torch::kFloat16 &&
                w.scalar_type() == torch::kFloat16, "fp16 CUDA tensors");
    TORCH_CHECK(x.numel() == 256 && w.numel() == 256, "probe takes exactly 16x16 inputs");
    auto xc = x.contiguous();
    auto wc = w.contiguous();
    auto y = torch::zeros({16, 16}, xc.options());
    const at::cuda::OptionalCUDAGuard guard(device_of(xc));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto px = reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
    auto pw = reinterpret_cast<const __half*>(wc.data_ptr<at::Half>());
    auto py = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
    const bool a_col = mode & 1, b_col = mode & 2, s_col = mode & 4, sep = mode & 8;
    if (!a_col && !b_col)
        hmma_probe_kernel<wmma::row_major, wmma::row_major><<<1, 1, 0, stream>>>(px, pw, py, s_col, sep);
    else if (a_col && !b_col)
        hmma_probe_kernel<wmma::col_major, wmma::row_major><<<1, 1, 0, stream>>>(px, pw, py, s_col, sep);
    else if (!a_col && b_col)
        hmma_probe_kernel<wmma::row_major, wmma::col_major><<<1, 1, 0, stream>>>(px, pw, py, s_col, sep);
    else
        hmma_probe_kernel<wmma::col_major, wmma::col_major><<<1, 1, 0, stream>>>(px, pw, py, s_col, sep);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hmma_matmul_f16", &hmma_matmul_f16,
          "Attempted fp16 X.W^T on tensor cores (does not compute it; see header)",
          pybind11::arg("x"), pybind11::arg("w"));
    m.def("hmma_layout_probe", &hmma_layout_probe,
          "Run one 16x16x16 tile under a chosen fragment interpretation",
          pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("mode"));
}
