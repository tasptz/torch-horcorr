#include <torch/extension.h>

void runKernelForward(
    uint32_t numBlocks, uint32_t blockSize,
    const float *left, const float *right, float *output,
    uint32_t leftWidth);

void runKernelRightBackward(
    uint32_t numBlocks, uint32_t blockSize,
    const float *left, const float *grad, float *output,
    uint32_t leftWidth);

void runKernelLeftBackward(
    uint32_t numBlocks, uint32_t blockSize,
    const float *right, const float *grad, float *output,
    uint32_t rightWidth);

torch::Tensor forward(at::Tensor left, at::Tensor right) {
    left = left.permute({0, 2, 3, 1}).contiguous();
    const auto batch = left.size(0);
    const auto height = left.size(1);
    const auto leftWidth = left.size(2);

    right = right.permute({0, 2, 3, 1}).contiguous();
    const auto rightWidth = right.size(2);
    auto n = leftWidth - rightWidth + 1;

    auto output = at::empty({batch, height, rightWidth, NWINDOWS}, left.options());

    runKernelForward(
        batch * height / BLOCKSIZE,
        BLOCKSIZE,
        left.data<float>(), right.data<float>(), output.data<float>(),
        leftWidth);

    return output.permute({0, 3, 1, 2});
}

torch::Tensor backwardRight(at::Tensor left, at::Tensor grad) {
    left = left.permute({0, 2, 3, 1}).contiguous();
    const auto batch = left.size(0);
    const auto height = left.size(1);
    const auto leftWidth = left.size(2);
    const auto rightWidth = leftWidth - NWINDOWS + 1;

    auto output = at::empty({batch, height, rightWidth, DIM}, left.options()).contiguous();

    grad = grad.permute({0, 2, 3, 1}).contiguous();

    runKernelRightBackward(
        batch * height / BLOCKSIZE,
        BLOCKSIZE,
        left.data<float>(),
        grad.data<float>(),
        output.data<float>(),
        leftWidth);

    return output.permute({0, 3, 1, 2});
}

torch::Tensor backwardLeft(at::Tensor right, at::Tensor grad) {
    right = right.permute({0, 2, 3, 1}).contiguous();
    const auto batch = right.size(0);
    const auto height = right.size(1);
    const auto rightWidth = right.size(2);
    const auto leftWidth = rightWidth + NWINDOWS - 1;

    auto output = at::empty({batch, height, leftWidth, DIM}, right.options()).contiguous();

    grad = grad.permute({0, 2, 3, 1}).contiguous();

    runKernelLeftBackward(
        batch * height / BLOCKSIZE,
        BLOCKSIZE,
        right.data<float>(),
        grad.data<float>(),
        output.data<float>(),
        rightWidth);

    return output.permute({0, 3, 1, 2});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Horizontal correlation", pybind11::return_value_policy::take_ownership)
    .def("backward_right", &backwardRight, "Horizontal correlation, right gradient", pybind11::return_value_policy::take_ownership);
    m.def("backward_left", &backwardLeft, "Horizontal correlation, left gradient", pybind11::return_value_policy::take_ownership);
    m.def("channels", [] { return DIM; }, "Number of channels");
    m.def("n_windows", [] { return NWINDOWS; }, "Number of windows");
    m.def("block_size", [] { return BLOCKSIZE; }, "Cuda block size");
}