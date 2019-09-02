#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

void runKernelForward(
    uint32_t numBlocks, uint32_t blockSize,
    const float *left, const float *right, float *output,
    uint32_t leftWidth,
    cudaStream_t stream);

void runKernelRightBackward(
    uint32_t numBlocks, uint32_t blockSize,
    const float *left, const float *grad, float *output,
    uint32_t leftWidth,
    cudaStream_t stream);

void runKernelLeftBackward(
    uint32_t numBlocks, uint32_t blockSize,
    const float *right, const float *grad, float *output,
    uint32_t rightWidth,
    cudaStream_t stream);

void forward(at::Tensor left, at::Tensor right, at::Tensor output) {
    const auto batch = left.size(0);
    const auto height = left.size(1);
    const auto leftWidth = left.size(2);

    runKernelForward(
        batch * height / BLOCKSIZE,
        BLOCKSIZE,
        left.data<float>(), right.data<float>(), output.data<float>(),
        leftWidth,
        at::cuda::getCurrentCUDAStream());
}

void backwardRight(at::Tensor left, at::Tensor grad, at::Tensor output) {
    const auto batch = left.size(0);
    const auto height = left.size(1);
    const auto leftWidth = left.size(2);

    runKernelRightBackward(
        batch * height / BLOCKSIZE,
        BLOCKSIZE,
        left.data<float>(),
        grad.data<float>(),
        output.data<float>(),
        leftWidth,
        at::cuda::getCurrentCUDAStream());
}

void backwardLeft(at::Tensor right, at::Tensor grad, at::Tensor output) {
    const auto batch = right.size(0);
    const auto height = right.size(1);
    const auto rightWidth = right.size(2);

    runKernelLeftBackward(
        batch * height / BLOCKSIZE,
        BLOCKSIZE,
        right.data<float>(),
        grad.data<float>(),
        output.data<float>(),
        rightWidth,
        at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Horizontal correlation")
    .def("backward_right", &backwardRight, "Horizontal correlation, right gradient")
    .def("backward_left", &backwardLeft, "Horizontal correlation, left gradient")
    .def("channels", [] { return DIM; }, "Number of channels")
    .def("n_windows", [] { return NWINDOWS; }, "Number of windows")
    .def("block_size", [] { return BLOCKSIZE; }, "Cuda block size");
}