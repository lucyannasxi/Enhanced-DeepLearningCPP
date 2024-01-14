#ifndef GRAPHDL_CORE_LAYERS_MATMUL_H_
#define GRAPHDL_CORE_LAYERS_MATMUL_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class MatmulLayer : public DifferentiableLayer
{
  public:
    MatmulLayer(ID id, const Tensor::SPtr& m1, const Tensor::SPtr& m2);

    TensorMap gradients(Tensor::SPtr output, Tensor::SPtr outputGrad) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::v