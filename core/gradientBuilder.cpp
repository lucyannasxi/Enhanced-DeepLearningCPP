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
    if (tensor-