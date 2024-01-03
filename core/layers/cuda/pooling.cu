#include "layers/cuda/macros.h"
#include "layers/cuda/utils.h"
#include "layers/pooling.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
namespace
{
// shapeParams = [inShape, outShape, kernelShape, strides]
__constant__ int shapeParams[12];

#define IN_SHAPE shapeParams
#define OUT_SHAPE (shapeParams + 4)
#define kernelX (shapeParams[8])
#define kernelY (shapeParams[9])
#define strideX (shapeParams[10])
#define strideY (shapeParams[11])

template <PaddingType padding>
__global__ void pool2D_max_nhwc_kernel(const float* in, float* out)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = n % shapeParams[7];
    n /= shapeParams[7];

    if (n < shapeParams[4] && x_out < shapeParams[5] &&
        y_out < shapeP