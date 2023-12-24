#include "constant.h"

#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"
#include "memory.h"

namespace graphdl
{
namespace core
{
namespace layers
{
ConstantLayer::ConstantLayer(ID id, float value, const TensorShape& shape,
                             MemoryType type)
    : Layer(id, {}, {createTensor(""