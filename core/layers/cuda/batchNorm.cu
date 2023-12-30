#include "layers/batchNorm.h"
#include "layers/elementwise.h"
#include "reduceUtils.cuh"
#include "utils.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
namespace
{
#define EPS 10e-8

__global__ void normalizeKernel(const float* alpha, const float* beta,
                                const float* stddev, float* y,
                                size_t featureSize, size_t size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        int id2 = id % featureSize;
        y[id] = alpha[id2] * y[id] / sqrt(stddev[id2] + EPS) + beta[id2];
    }
}

__global__ void alphaGradKernel(const float* bet