#include "reshape.h"

#include "abstractTensor.h"
#ifdef CUDA_AVAILABLE
#include "cuda/utils.h"
#endif
#include "graph.h"
#include "layer.h"

#include <cstring>

namespace graphdl
{
namespace core
{
namespace layers
{
ReshapeLayer::ReshapeLayer(ID id, const Tensor::SPtr& t,
                           const TensorShape& shape)
    : DifferentiableLayer(id, {t}, {createTensor("", shape, t->getType())})
{
}

Layer::TensorMap ReshapeLayer::gradients(Tensor::SPtr /* out */,
                                         T