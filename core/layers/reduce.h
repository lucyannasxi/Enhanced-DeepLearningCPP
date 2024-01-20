#ifndef GRAPHDL_CORE_LAYERS_REDUCE_SUM_H_
#define GRAPHDL_CORE_LAYERS_REDUCE_SUM_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
enum class ReduceType
{
    kSUM = 0,
    kMAX = 1,
    kMIN = 2,
};

class ReduceBackLayer : public DifferentiableLayer
{
  public:
    ReduceBackLayer(ID id, const Tensor::SPtr& tensor, int numAxes,
                    ReduceType reduceType);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    int mNumAxes;
    ReduceType mReduceType;
};

class ReduceBackGradientLayer : public Layer
{
  public:
    ReduceBackGradientLayer(ID id, const Tensor::SPtr& in,
                            const Tensor::SPtr& out,
                            const Tensor::SPtr& outGrad, int numAxes,
                            ReduceType reduceType);

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    int mNumAxes;
    ReduceType mReduceType;
};

class ReduceFrontLayer : public DifferentiableLayer
{
  public:
    ReduceFrontLayer(ID id, const Tensor::SPtr& tensor, int numAxes,
                     ReduceType reduceType);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    int mNumAxes;
    ReduceType mReduceType;
};

class ReduceFrontGradientLayer : public Layer
{
  public:
    ReduceFrontGradientLayer(ID id, const Tensor::SPtr& in,
                             const Tensor::SPtr& out,
                             const Tensor::SPtr& outGrad, int numAxes,
                             ReduceType reduceType);

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    int mNumAxes;
    ReduceType mReduceType;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
void runReduceBackDevice(const float* in, float* out, size_t outSize,
                         size_t reduceSize, ReduceType reduceType);

void runReduceBackGradientDevice(const float* in, const float* out,
                                 const float* outGrad, float* inGrad,
                                 size_t outSize, size_t reduceSize,
           