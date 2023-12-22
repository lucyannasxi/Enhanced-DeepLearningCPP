#include "uniformInitializer.h"

#ifdef CUDA_AVAILABLE
#include "layers/cuda/randomUtils.h"
#endif
#include "abstractInitializer.h"

#include <random>

namespace graphdl
{
namespace core
{
namespace initializers
{
UniformInitializer::UniformInitializer(float min, float max, size_t seed)
    : Initializer(seed), mMinValue(min), mMaxValue(max)
{
}

void UniformInitializer::initHost(float* memory, const TensorShape& shape)
{
    std::uniform_real_distribution<> d(mM