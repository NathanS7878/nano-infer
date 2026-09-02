// Python bindings for the nano-infer CUDA kernels.
//
// Kernel 1 originally carried PYBIND11_MODULE inside rmsnorm.cu, which works
// only while there is exactly one .cu file. Adding kernel 2 forced the split:
// each .cu now defines its op and nothing else, and this single translation
// unit owns the module. Kernel 4 (decode attention) just adds a
// declaration and a def() line here.

#include <torch/extension.h>

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps);
torch::Tensor swiglu_forward(torch::Tensor gate, torch::Tensor up);
torch::Tensor rope_forward(torch::Tensor x, torch::Tensor cos, torch::Tensor sin);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward,
          "Fused RMSNorm (CUDA)",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("eps"));
    m.def("swiglu_forward", &swiglu_forward,
          "Fused SwiGLU: silu(gate) * up (CUDA)",
          pybind11::arg("gate"), pybind11::arg("up"));
    m.def("rope_forward", &rope_forward,
          "Fused RoPE: x*cos + rotate_half(x)*sin (CUDA)",
          pybind11::arg("x"), pybind11::arg("cos"), pybind11::arg("sin"));
}
