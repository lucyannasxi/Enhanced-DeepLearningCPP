#ifndef GRAPHDL_CORE_LAYERS_BATCH_NORM_H_
#define GRAPHDL_CORE_LAYERS_BATCH_NORM_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class BatchNormLayer : public DifferentiableLayer
{
  public:
    BatchNormLayer(ID id, const Tensor::SPtr& tensor, const Tensor::SPtr& alpha,
                   const Tensor::SPtr& beta, int numAxes);

    void initialize() override;

    ~BatchNormLayer();

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    int mNumAxes;
    Memory<float> mMean;
    Memory<float> mStddev;
};

class BatchNormGradientLayer : public Layer
{
  public:
    BatchNormGradientLayer(ID id, const Tensor::SPtr& tensor,
                           const Tensor::SPtr& alpha, const Tensor::SPtr& beta,
                           const Tensor::SPtr& out, const Tensor::SPtr& outGrad,
                           int numAxes, Memory<float>* mean,
          