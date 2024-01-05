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

//////////////////////////////////////////////////////////////