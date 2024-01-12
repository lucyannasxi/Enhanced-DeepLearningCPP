#ifndef GRAPHDL_CORE_LAYERS_ELEMENTWISE_H_
#define GRAPHDL_CORE_LAYERS_ELEMENTWISE_H_

#include "differentiableLayer.h"

#include <functional>

namespace graphdl
{
namespace core
{
namespace layers
{
enum class Elementwise : int
{
    kADD = 0,
    kSUB = 1,
    kMUL = 2,
    kDIV = 3
};

using ElementwiseFun = std::function<float(float, float)>;

class ElementwiseBackLayer : public DifferentiableLayer
{
  public:
    ElementwiseBackLayer(ID id, const Tensor::SPtr& t1, const Tensor::SPtr& t2,
                         Elementwise op);

    TensorMap gradients(Tensor::SPtr output, Tensor::SPtr outputGrad) override;

  private:
    void execute(const s