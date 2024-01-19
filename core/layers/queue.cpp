#include "queue.h"

#include "abstractTensor.h"
#include "graph.h"

namespace graphdl
{
namespace core
{
namespace layers
{
QueueLayer::QueueLayer(ID id, const std::vector<Tensor::SPtr>& ops)
    : Layer(id, o