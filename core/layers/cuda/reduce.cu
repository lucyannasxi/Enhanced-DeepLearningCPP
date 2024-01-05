#include "layers/reduce.h"
#include "reduceUtils.cuh"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
void runReduceBackDevice(const float* x, float* y, size_t outSize,
                         size_t reduceSize, ReduceType reduceType)
{
    switch (reduceType)
    {
    case ReduceType::kSUM:
        reduce<ReduceOpCuda::kSUM>(x, y, outSize, reduceSize);
        break;
    case ReduceType::kMIN:
        reduce<ReduceOpCuda::kMIN>(x, y, outSize, reduceSize);
        break;
    case ReduceType::kMAX:
        reduce<ReduceOpCuda::kMAX>(x, y, outSize, reduceSize);
        break;
    }
}

void runReduceBackGradientDevice(const float* x, const float* y,
                                 const float* yGrad, float* xGrad,
                                 size_t outSize, size_t reduceSize,
                                 ReduceType reduceType)
{
    switch (reduceType)
    {
    case ReduceType::kSUM:
        reduceGradient<ReduceOpCuda::kSUM>(x, y, yGrad, xGrad, outSize,
                                           reduceSize);
        break;
    case ReduceType::kMIN:
        reduceGradient<ReduceOpCuda::kMIN>(x, y, yGrad, xGrad, outSize,
                                           reduceSize);
        break;
    case ReduceType::kMAX:
        reduceGradient<ReduceOpCuda::kMAX>(x, y, yGrad, xGrad, outSize,
                       