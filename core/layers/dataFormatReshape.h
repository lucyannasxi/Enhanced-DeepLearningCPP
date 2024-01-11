#ifndef GRAPHDL_CORE_LAYERS_DATA_FORMAT_RESHAPE_H_
#define GRAPHDL_CORE_LAYERS_DATA_FORMAT_RESHAPE_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class Nhwc2NchwLayer : public DifferentiableLayer
{
  public:
    Nhwc2NchwLayer(ID id, const Tensor::SPtr& tensor);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

  private:
    void execute(const std::vector