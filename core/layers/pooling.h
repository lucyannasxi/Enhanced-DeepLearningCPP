#ifndef GRAPHDL_CORE_LAYERS_POOLING_H_
#define GRAPHDL_CORE_LAYERS_POOLING_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
enum class PoolingType
{
    kMAX = 0,
    kAVERAGE = 1
};

enum class PaddingType
{
    kVALID = 0,
    kSAME = 1
};

enum class DataFormat
{
    kNHWC = 0,
    kNCHW = 1
};

PaddingType str2padding(const std::s