#include <stdint.h>

template <typename T>
inline const T &min(const T &a, const T &b) {
    return a <= b ? a : b;
}

__global__ void kernelForward(
    const float *left,
    const float *right,
    float *output,
    uint32_t leftWidth) {

    const int row = threadIdx.x + blockIdx.x * blockDim.x;
    const int indexLeft = row * leftWidth * DIM;
    const int rightWidth = leftWidth - NWINDOWS + 1;
    const int indexRight = row * rightWidth * DIM;
    const int indexOut = row * rightWidth * NWINDOWS;
    float *pOutput = output + indexOut;

    float vecLeft[NWINDOWS][DIM];
    const float *pLeft = left + indexLeft;
    memcpy(vecLeft, pLeft, NWINDOWS * DIM * sizeof(float));
    pLeft += NWINDOWS * DIM;

    float vecRight[DIM];
    const float *pRight = right + indexRight;

    int idxRing = 0;
    for (int w = 0; w < rightWidth; ++w) {
        memcpy(vecRight, pRight, DIM * sizeof(float));
        pRight += DIM;

        if (w > 0) {
            memcpy(vecLeft[idxRing], pLeft, DIM * sizeof(float));
            idxRing = (idxRing + 1) % NWINDOWS;
            pLeft += DIM;
        }

        #pragma unroll
        for (int i = 0; i < NWINDOWS; ++i) {
            const int idx = (idxRing + i) % NWINDOWS;
            float sum = 0.f;

            #pragma unroll
            for (int j = 0; j < DIM; ++j) {
                sum += vecLeft[idx][j] * vecRight[j];
            }
            *pOutput++ = sum;
        }
    }
}

void runKernelForward(
    uint32_t numBlocks, uint32_t blockSize,
    const float *left, const float *right, float *output,
    uint32_t leftWidth) {

    kernelForward<<<numBlocks, blockSize>>>(left, right, output, leftWidth);
}

__global__ void kernelRightBackward(
    const float *left,
    const float *grad,
    float *output,
    uint32_t leftWidth) {

    const int row = threadIdx.x + blockIdx.x * blockDim.x;
    const int indexLeft = row * leftWidth * DIM;
    const float *pLeft = left + indexLeft;
    const int rightWidth = leftWidth - NWINDOWS + 1;
    const int indexGrad = row * rightWidth * NWINDOWS;
    const float *pGrad = grad + indexGrad;
    const int indexOut = row * rightWidth * DIM;
    float *pOutput = output + indexOut;

    for (int w = 0; w < rightWidth; ++w) {
        #pragma unroll
        for (int d = 0; d < DIM; ++d) {
            float s = 0.f;
            #pragma unroll
            for (int n = 0; n < NWINDOWS; ++n) {
                s += pLeft[d + n * DIM] * pGrad[n];
            }
            *pOutput++ = s;
        }
        pLeft += DIM;
        pGrad += NWINDOWS;
    }
}

void runKernelRightBackward(
    uint32_t numBlocks, uint32_t blockSize,
    const float *left, const float *grad, float *output,
    uint32_t leftWidth) {
    kernelRightBackward<<<numBlocks, blockSize>>>(left, grad, output, leftWidth);
}

__global__ void kernelLeftBackward(
    const float *right,
    const float *grad,
    float *output,
    uint32_t rightWidth) {

    const int row = threadIdx.x + blockIdx.x * blockDim.x;
    const int indexRight = row * rightWidth * DIM;
    const float *pRight = right + indexRight;
    const float *const pRightEnd = pRight + rightWidth * DIM;
    const int leftWidth = rightWidth + NWINDOWS - 1;
    const int indexGrad = row * rightWidth * NWINDOWS;
    const float *const pGradBegin = grad + indexGrad;
    const int indexOut = row * leftWidth * DIM;
    float *pOutput = output + indexOut;

    for (int w = 0; w < leftWidth; ++w) {
        #pragma unroll
        for (int d = 0; d < DIM; ++d) {
            float s = 0.f;
            #pragma unroll
            for (int n = 0; n < min(w + 1, NWINDOWS); ++n) {
                const float *p = pRight + (d - n * DIM);
                s += p < pRightEnd ? (*p * pGradBegin[(w - n) * NWINDOWS + n]) : 0.f;
            }
            *pOutput++ = s;
        }
        pRight += DIM;
    }
}

void runKernelLeftBackward(
    uint32_t numBlocks, uint32_t blockSize,
    const float *right, const float *grad, float *output,
    uint32_t rightWidth) {
    kernelLeftBackward<<<numBlocks, blockSize>>>(right, grad, output, rightWidth);
}