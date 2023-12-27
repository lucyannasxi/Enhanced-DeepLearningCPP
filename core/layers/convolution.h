#ifndef GRAPHDL_CORE_LAYERS_CONVOLUTION_H_
#define GRAPHDL_CORE_LAYERS_CONVOLUTION_H_

#include "differentiableLayer.h"
#include "pooling.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class Conv2DLayer : public DifferentiableLayer
{
  public:
    Conv2DLayer(ID id, const Tensor::SPtr& t, const Tensor::SPtr& kernel,
                const std::vector<int>& strides, PaddingType padding,
                DataFormat dataFormat);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

    void initialize() override;

    ~Conv2DLayer();

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& in