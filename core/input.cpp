#include "input.h"

#include <cstring>

namespace graphdl
{
namespace core
{
InputLayer::InputLayer(ID id, const std::string& name, const Shape& shape,
                       MemoryType type)
    : Layer(id, {}, {createTensor(name, shape,