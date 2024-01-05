#ifndef GRAPHDL_CORE_LAYERS_CUDA_REDUCE_UTILS_H_
#define GRAPHDL_CORE_LAYERS_CUDA_REDUCE_UTILS_H_

#include <cuda.h>
#include <cfloat>
#include "utils.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{

enum class ReduceOpCuda
{
    ///< sum of elements
    kSUM = 0,

    ///< sum of elements squared
    kSQUARED_SUM = 1,

    kMIN = 2,
    kMAX = 3,

    ///< mean of elements
    kMEAN = 4,

    ///< mean of elements squared
    kSQUARED_MEAN = 5
};

enum class ReduceBinOpCuda
{
    ///< each reduction does dot product
    kDOT_PRODUCT = 0,
};

namespace
{

///////////////////////////////////////////////////////////////
// Unary reduction
///////////////////////////////////////////////////////////////

// represent initial value from which reduction starts
// (initial value for accumulator)
template <ReduceOpCuda op> __device__ float initialValue()
{
    return 0.;
}
template <> __device__ float initialValue<ReduceOpCuda::kMIN>()
{
    return FLT_MAX;
}
template <> __device__ float initialValue<ReduceOpCuda::kMAX>()
{
    return -FLT_MAX;
}

// represents initial transformation for elements
template <ReduceOpCuda op>
__device__ float initialReduceOp(float x)
{
    return x;
}
template <>
__device__ float initialReduceOp<ReduceOpCuda::kSQUARED_SUM>(float x)
{
    return x * x;
}
template <>
__device__ float initialReduceOp<ReduceOpCuda::kSQUARED_MEAN>(float x)
{
    return x * x;
}

// describes how to reduce elements
template <ReduceOpCuda op> __device__ float reduceOp(float f1, float f2)
{
    return f1 + f2;
}
template <> __device__ float reduceOp<ReduceOpCuda::kMIN>(float f1, float f2)
{
    return f1 < f2 ? f1 : f2;
}
template <> __device__ float reduceOp<ReduceOpCuda::kMAX>(float f1, float f2)
{
    return f1 > f2 ? f1 : f2;
}

// describes how to compute gradient
template <ReduceOpCuda op> __device__ float reduceGradientOp(float x, float y);
template <>
__device__ float reduceGradientOp<ReduceOpCuda::kSUM>(float x, float y)
{
    return 1.;
}
template <>
__device__ float reduceGradientOp<ReduceOpCuda::kMEAN>(float x, float y)
{
    return 1.;
}
template <>
__device__ float reduceGradientOp<ReduceOpCuda::kSQUARED_SUM>(float x, float y)
{
    return 2 * x;
}
template <>
__device__ float reduceGradientOp<ReduceOpCuda::kSQUARED_MEAN>(float x, float y)
{
    return 2 * x;
}
template <>
__device__ float reduceGradientOp<ReduceOpCuda::kMIN>(float x, float y)
{
    return float(x == y);
}
template <>
__device__ float reduceGradientOp<ReduceOpCuda::kMAX>(float x, float y)
{
    return float(x == y);
}

template <ReduceOpCuda op, unsigned BS>
__device__ void warpReduce(volatile float *sdata, unsigned tid)
{
    if (BS >= 64) sdata[tid] = reduceOp<op>(sdata[tid], sdata[tid + 32]);
    if (BS >= 32) sdata[tid] = reduceOp<op>(sdata[tid], sdata[tid + 16]);
    if (BS >= 16) sdata[tid]