#include "batchNorm.h"

#include "abstractTensor.h"
#include "activation.h"
#include "elementwise.h"
#include "graph.h"
#include "graphdl_ops.h"
#include "reduce.h"

#include <cmath>

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
#define EPS 10e-8

size_t getBatchSize(const Tensor::SPtr& tensor, int numAxes)
{
    TensorShape shape = tensor->getShape();
    size_t size = 1;
    for (int i = 0; i < numAxes; ++i) size *= shape[i];
    return size;
}

size_t getFeatureSize(const Tensor::SPtr& tensor, int numAxes)
{
    return tensor->getShape().getCount() / getBatchSize(tensor, numAxes);
}

}  // namespace

void runBatchNormHost(const float* x, const float* alpha, const float* beta,
                      float* y, float* mean, float* stddev, size_t size,
                      size_t batchSize)
{
    size_t stride = size / batchSize;

    for (int i = 0; i < stride; ++i)
    {
        float val = 0.;
        for (int j = 0; j < batchSize; ++j) val += x[j * stride + i];
        mean[i] = val / float(batchSize);
    }

    for (int i = 0; i < stride; ++i)
    {
        float val = 0.;
        for (int j = 0; j < batchSize; ++j)
        {
            float f = x[j * stride + i] - mean[i];
            val += f * f;
        }
        stddev[i] = val / float(batchSize);
    }

    for (int i = 0; i < stride; ++i)
    {
        for (int j = 0; j < batchSize; ++j)
        {
            float val = x[j * stride + i] - mean[i];
            val /= std::sqrt(stddev[i] + EPS);
            val *= alpha[i];
            val += beta[i];
            y[j * stride + i] = val;
        }
    }
}

void runBatchNormGradientHost(const float* x, const float* alpha,
                              const float* beta, const float* y,
                              const float* yGrad, const float* mean,
                              const float* stddev, float* xGrad,
                              float* alphaGrad, float* betaGrad, size_t size,
                              size_t batchSize)
{
    size_t stride = size / batchSize;

    // betaGrad
    for (int i = 0; i < stride; ++i)
    {
        float val = 0.;
        for (int j = i; j < size; j += stride) val += yGrad[j];
        betaGrad[i] = val;
    }

    // alphaGrad
    for (int i = 0; i < stride; ++i)
    {
        float val = 0.;
        for (int j = i; j < size; j += stride) val += yGrad[j] * y[j];
        alphaGrad[i] = (val - betaGrad[i] * beta[i]) / alpha[i];
    }

    // xGrad
    for (int i = 0; i < stride; ++i)
    {
        float val = -betaGrad[i] * mean[i];
        for (int j = i; j < size; j += stride) val += yGrad[j] * x[j];

        for (int j = i; j < size; j += stride)
        {
            xGrad[j] = yGrad[j] - betaGrad[i] / float(batchSize) -
                       0.5 * (x[j] - mean[i]) * val / (stddev[i] + EPS);
            xGrad[j] /= std::sqrt(stddev[i] + EPS);
            xGrad[j] *= alpha[i];
        }
    }
}

BatchNormLayer::BatchNormLayer(ID id, const Tensor::SPtr& tensor,
                               const Tensor::SPtr& alpha,
                               const Tensor::SPtr& beta, int numAxes)
    : DifferentiableLayer(
          id, {tensor, alpha, beta},
          {createTensor("", tensor->getShape(), tensor->getType())}),
      mNumAxes(numAxes),
      mMean(tensor->getType(), getFeatureSize(tensor, 