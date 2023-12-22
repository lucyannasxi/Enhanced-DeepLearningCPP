#include "layer.h"

#include "graph.h"

#include <utility>

namespace graphdl
{
namespace core
{
Layer::Layer(ID id, const std::vector<Tensor::SPtr>& inputs,
             std::vector<Tensor::SPtr> outputs)
    : mID(id), mIsEvaluated(false), mOutputs(std::move(out