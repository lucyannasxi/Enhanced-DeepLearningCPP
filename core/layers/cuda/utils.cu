#include "utils.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
namespace utils
{
namespace
{
__global__ void fillKernel(float* memory, size_t size, float value)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) memory[id] = value;
}

template <Elementwise elem>
__device__ float op(float f1, float f2);
template <>
__device__ float op<Elementwise::kADD>(float f1, float f2)
{
    return f1 + f2;
}
template <>
__device__ float op<Elementwise::kSUB>(float f1, float f2)
{
    return f1 - f2;
}
template <>
__device__ float op<Elementwise::kMUL>(float f1, float f2)
{
    return f1 * f2;
}
template <>
__device__ float op<Elementwise::kDIV>(float f1, float f2)
{
    return f1 / f2;
}

template <Elementwise elem, int b>
__global__ void elementwiseCastFrontKernel(const float* x1, size_t size,
                                           const float* x2, size_t reduceSize,
                                           float* y)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        if (b == 1)
            y[id] = op<elem>(x1[id], 