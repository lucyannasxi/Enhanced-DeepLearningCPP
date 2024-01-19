#ifndef GRAPHDL_CORE_LAYERS_REDUCE_SUM_H_
#define GRAPHDL_CORE_LAYERS_REDUCE_SUM_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
enum class ReduceType
{
    kSUM = 0,
    kMAX = 1,
    kMIN = 2,
};

class ReduceBackLayer : public DifferentiableLayer
{
  public:
    ReduceBackLayer(ID id, const Tensor::SPtr& tensor, int numAxes,
                    ReduceType reduceType);

    TensorMap gradients(Tensor