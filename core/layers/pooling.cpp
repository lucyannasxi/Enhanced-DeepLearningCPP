#include "pooling.h"

#include "abstractTensor.h"
#include "graph.h"
#include "graphdl.h"
#include "graphdl_ops.h"
#include "pooling_host.h"

#include <cassert>
#include <utility>

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
Tensor::SPtr createPoolingOutput(const Tensor::SPtr& t,
                                 const std::vector<int>& kernel,
                                 const std::vector<int>& strides,
                                 PaddingType padding, DataFormat dataFormat)
{
    TensorShape shape = t->getShape();

    if (dataFormat == DataFormat::kNHWC)
    {
        if (padding == PaddingType::kVALID)
        {
            shape[1] = ceil(shape[1] - kernel[0] + 1, strides[0]);
            shape[2] = ceil(shape[2] - kernel[1] + 1, strides[1]);
        }
        else  // padding == PaddingType::kSAME
        {
    