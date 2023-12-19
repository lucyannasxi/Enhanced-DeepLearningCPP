#include "gradientBuilder.h"

#include "abstractTensor.h"
#include "addN.h"
#include "constant.h"
#include "differentiableLayer.h"
#include "elementwise.h"
#include "graph.h"
#include "graphdl_ops.h"

namespace graphdl
{
namespace core
{
GradientBuilder::GradientBuilder(const Tensor::SPtr& tensor,
                                 const std::vector<Tensor::SPtr>& weights)
    : mTensor(tensor), mWeights(weights.begin(), weights.end())
{
    if (tensor->getShape().getCount() != 1)
        throw std::runtime_error("Not scalar gradient calculation");
}

void GradientBuilder::findTensorOutputs(const Tensor::SPtr& tensor,
                                        std::set<Tensor::SPtr>& visited)
{
    if (visited.count(tensor) > 0) return;
    visited.insert(tensor);

    Layer::SPtr layer = tensor->getLayer();
    if (layer->hasGradient())
    {
        std::vector<Tensor::SPtr> inputs = layer->getInputs();
        for (Tensor::SPtr in : inputs)
        {
            if (mGradientsToCalc.count(in) == 0)
                mGradientsToCalc.insert({in, {}});
            mGradientsToCalc[in].insert(tensor);
            findTensorOutputs(in, visited);
        }
    }
}

GradientBuilder::TensorMap GradientBuilder::createGradients()
{
    std::set<Tensor::SPtr> visited;
    findTensorOutputs(mTensor, visited);

    mTensorGradients.clear();
    mCalculatedTensors.clear();
    //! d(mTensor)/d(mTensor) = 1.
    mCalculatedTensors.insert(
        {mTensor, constant(1., mTensor->getShape(), mTensor->getType())});

    calculateGradientsForTensor(mTensor);

    TensorMap gradients;
    for (const Tensor::SPtr& weights : mWeights)
    {
        if (mCalculatedTensors.count(weights) == 0)
        {
            Tensor::SPtr nullGradient =
                constant(0., weights->getShape(), weights->getType());
            mCalculatedTensors.insert({weights, nullGradient});
        }
        gradients.insert({weights, mCalculatedTensors[weights]});
    }
    return gradients;
}

void GradientBuilder::modifyTensorGradient(const Tensor::SPtr& tensor,
                                           const Tensor::SPtr& tensorGrad)
{
    if (mTensorGradients.count(tensor) == 0)
        mTensorGradients.insert({tensor, {}});
    mTensorGradients[tensor].push_back(tensorGrad);

    if (mGradientsToCalc[tensor].empty())
        mCalculatedTensors[tensor] = core::addN(mTensorGradients[tensor]);
}

void GradientBuilder::calculateGradientsForTensor(const Tensor::SPtr& tensor)
{
    if (!mGradientsToCalc[tensor].empty()) return;
    Tensor::SPtr tensorGrad = mCalculatedTensors[tensor];

    Layer::SPtr layer = tensor->getLayer();
    if (layer->hasGradient())
    {
        std::vector<Tensor::SPtr> input