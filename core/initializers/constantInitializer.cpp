#include "constantInitializer.h"

#include "abstractInitializer.h"
#include "graphdl.h"
#ifdef CUDA_AVAILABLE
#include "layers/cuda/utils.h"
#endif

#include <algorithm>

namespace graphdl
{
namespace core
{
namespace initializers
{
ConstantInitializer::ConstantInitializer(float value)
    : Initializer(0), mValue(value)
{
}

void ConstantInitializer::initHost(float* memory, const TensorShape& shape)
{
    std::fill_n(memory, shape.get