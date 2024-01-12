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
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    Elementwise mOp;
    ElementwiseFun mFun;
};

class ElementwiseBackGradientLayer : public Layer
{
  public:
    ElementwiseBackGradientLayer(ID id, const Tensor::SPtr& t1,
                                 const Tensor::SPtr& t2, Tensor::SPtr out,
                                 Tensor::SPtr outGrad, Elementwise op);

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    Elementwise mOp;
    ElementwiseFun mFun1, mFun2;
};

class ElementwiseFrontLayer : public DifferentiableLayer
{
  public:
    ElementwiseFrontLayer(ID id, const Tensor::SPtr& t1, const Tensor::SPtr& t2,
                          Elementwise op);

    TensorMap gradients(Tensor::SPtr output, Tensor::SPtr outputGrad) override;

  private:
    void execute(const std::vect