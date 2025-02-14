
#include "layers/activation.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
template <Activation act>
__device__ float activation(float x);

template <>
__device__ float activation<Activation::kRELU>(float x)
{
    return x > 0. ? x : 0.;
}
template <>
__device__ float activation<Activation::kSIGMOID>(float x)
{
    return 1. / (1. + expf(-x));
}
template <>
__device__ float activation<Activation::kTANH>(float x)
{
    return tanhf(x);
}
template <>
__device__ float activation<Activation::kSQUARE>(float x)
{
    return x * x;
}
template <>
__device__ float activation<Activation::kABS>(float x)
{
    return x >= 0. ? x : -x;
}
template <>
__device__ float activation<Activation::kNEG>(float x)
{
    return -x;
}
template <>
__device__ float activation<Activation::kRECIPROCAL>(float x)
{
    return 1. / x;
}
template <>
__device__ float activation<Activation::kLOG>(float x)
{
    return logf(x);
}
template <>
__device__ float activation<Activation::kSQRT>(float x)
{
    return sqrtf(x);
}
template <>
__device__ float activation<Activation::kEXP>(float x)
{
    return expf(x);
}
template <>
__device__ float activation<Activation::kLEAKY_RELU>(float x)
{
    return x >= 0. ? x : 0.01 * x;
}
template <>
__device__ float activation<Activation::kRELU_6>(float x)
{
    return x >= 0. ? (x <= 6. ? x : 6.) : 0.;
}
template <>
__device__ float activation<Activation::kELU>(float x)
{
    return x >= 0. ? x : expf(x) - 1.;
}
template <>
__device__ float activation<Activation::kSOFTPLUS>(float x)
{
    return logf(expf(x) + 1.);
}
template <>
__device__ float activation<Activation::kSOFTSIGN>(float x)
{
    return x / ((x > 0. ? x : -x) + 1.);
}

template <Activation act>
__global__ void activationKernel(const float* x, float* y, size_t size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) y[id] = activation<act>(x[id]);
}

template <Activation act>
__device__ float activationGradient(float x, float y);

template <>
__device__ float activationGradient<Activation::kRELU>(float x, float y)
{
    return x >= 0. ? 1. : 0.;
}
template <>
__device__ float activationGradient<Activation::kSIGMOID>(float x, float y)
{
    return y * (1. - y);
}
template <>
__device__ float activationGradient<Activation::kTANH>(float x, float y)
{
    return 1. - y * y;
}
template <>
__device__ float activationGradient<Activation::kSQUARE>(float x, float y)
{
    return 2. * x;
}
template <>
__device__ float activationGradient<Activation::kABS>(float x, float y)
{
    return x >= 0. ? 1. : -1.;
}
template <>
__device__ float activationGradient<Activation::kNEG>(float x, float y)
{
    return -1;
}
template <>
__device__ float activationGradient<Activation::kRECIPROCAL>(float x, float y)
{
    return -1. * y * y;
}
template <>
__device__ float activationGradient<Activation::kLOG>(float x, float y)
{
    return 1. / x;
}
template <>
__device__ float activationGradient<Activation::kSQRT>(float x, float y)
{
    return 1. / (2. * y);
}
template <>
__device__ float activationGradient<Activation::kEXP>(float x, float y)
{
    return y;
}
template <>
__device__ float activationGradient<Activation::kLEAKY_RELU>(float x,
                                                             float /* y */)
{
    return x >= 0. ? 1. : 0.01;
}
template <>
__device__ float activationGradient<Activation::kRELU_6>(float x, float /* y */)
{
    return x >= 0. ? (x <= 6. ? 1. : 0.) : 0.;
}
template <>
__device__ float activationGradient<Activation::kELU>(float x, float y)
{
    return x >= 0. ? 1. : y + 1.;
}
template <>
__device__ float activationGradient<Activation::kSOFTPLUS>(float x,
                                                           float /* y */)
{
    float v = expf(x);
    return v / (v + 1.);
}
template <>
__device__ float activationGradient<Activation::kSOFTSIGN>(float x,
                                                           float /* y */)
{
    float v = (x > 0. ? x : -x) + 1.;
    return 1. / (v * v);
}

template <Activation act>
__global__ void activationGradientKernel(const float* x, const float* y,
                                         const float* yGrad, float* xGrad,
                                         size_t size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
        xGrad[id] = yGrad[id] * activationGradient<act>(x[id], y[id]);
}

void runActivationDevice(const float* x, float* y, size_t size, Activation op)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (op)
    {
    case Activation::kRELU:
        activationKernel<Activation::kRELU>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    case Activation::kSIGMOID:
        activationKernel<Activation::kSIGMOID>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    case Activation::kTANH:
        activationKernel<Activation::kTANH>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    case Activation::kSQUARE:
        activationKernel<Activation::kSQUARE>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    case Activation::kABS:
        activationKernel<Activation::kABS>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    case Activation::kNEG:
        activationKernel<Activation::kNEG>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    case Activation::kRECIPROCAL:
        activationKernel<Activation::kRECIPROCAL>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    case Activation::kLOG:
        activationKernel<Activation::kLOG>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    case Activation::kSQRT:
        activationKernel<Activation::kSQRT>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    case Activation::kEXP:
        activationKernel<Activation::kEXP>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    case Activation::kLEAKY_RELU:
        activationKernel<Activation::kLEAKY_RELU>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    case Activation::kRELU_6:
        activationKernel<Activation::kRELU_6>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    case Activation::kELU:
        activationKernel<Activation::kELU>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    case Activation::kSOFTPLUS:
        activationKernel<Activation::kSOFTPLUS>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    case Activation::kSOFTSIGN:
        activationKernel<Activation::kSOFTSIGN>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, size);
        break;
    }
}

void runActivationGradientDevice(const float* x, const float* y,
                                 const float* yGrad, float* xGrad, size_t size,
                                 Activation op)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (op)
    {
    case Activation::kRELU:
        activationGradientKernel<Activation::kRELU>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    case Activation::kSIGMOID:
        activationGradientKernel<Activation::kSIGMOID>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    case Activation::kTANH:
        activationGradientKernel<Activation::kTANH>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    case Activation::kSQUARE:
        activationGradientKernel<Activation::kSQUARE>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    case Activation::kABS:
        activationGradientKernel<Activation::kABS>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    case Activation::kNEG:
        activationGradientKernel<Activation::kNEG>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    case Activation::kRECIPROCAL:
        activationGradientKernel<Activation::kRECIPROCAL>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    case Activation::kLOG:
        activationGradientKernel<Activation::kLOG>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    case Activation::kSQRT:
        activationGradientKernel<Activation::kSQRT>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    case Activation::kEXP:
        activationGradientKernel<Activation::kEXP>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    case Activation::kLEAKY_RELU:
        activationGradientKernel<Activation::kLEAKY_RELU>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    case Activation::kRELU_6:
        activationGradientKernel<Activation::kRELU_6>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    case Activation::kELU:
        activationGradientKernel<Activation::kELU>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    case Activation::kSOFTPLUS:
        activationGradientKernel<Activation::kSOFTPLUS>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    case Activation::kSOFTSIGN:
        activationGradientKernel<Activation::kSOFTSIGN>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yGrad, xGrad, size);
        break;
    }
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl