// CPU-only pybind binding for the vendored FPS kernels (fps_cpu.cpp is
// vendored from torch-fps 0.5 minus its kNN kernel; device dispatch lives in Python).
#include <torch/extension.h>

#include <cstdint>

namespace torch_fps {

at::Tensor fps_forward_cpu(
    const at::Tensor& points,
    const at::Tensor& mask,
    const at::Tensor& start_idx,
    int64_t K);

}  // namespace torch_fps

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Release the GIL: bodies are pure ATen (multi-ms at::parallel_for loops
    // would otherwise block all Python threads).
    m.def(
        "fps_forward",
        &torch_fps::fps_forward_cpu,
        "Farthest point sampling forward pass (CPU)",
        pybind11::arg("points"),
        pybind11::arg("mask"),
        pybind11::arg("start_idx"),
        pybind11::arg("K"),
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
