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

__global__ void alphaGradKernel(const float* betaGrad, const float* beta,
                                const float* alpha, float* alphaGrad,
                                size_t featureSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < featureSize)
        alphaGrad[id] = (alphaGrad[id] - betaGrad[id] * beta[id]) / alpha[id];
}

__global__ void xGradKernel(const float* x, const float* alpha, const float* y,
                            const float* yGrad, const float* mean,
                            const float* stddev, const float* betaGrad,
                            float* xGrad, size_t featureSize, size_t batchSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < featureSize * batchSize)
    {
        int c = id / batchSize;
        int pos = (id - c * batchSize) * featureSize + c;

        float val = -betaGrad[c] * mean[c];
        for (int i = c; i < featureSize * batchSize; i += featureSize)
            val += yGrad[i] * x[i];

        float out = yGrad[pos] - betaGrad[c] / float(batchS