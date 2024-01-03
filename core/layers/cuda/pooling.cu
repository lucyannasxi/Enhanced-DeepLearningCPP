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
        y_out < shapeParams[6] && c < shapeParams[7])
    {
        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (kernelX - 1) / 2;
            y_in -= (kernelY - 1) / 2;
        }

        float val = 0.;
        if (x_in >= 0 && y_in >= 0)
            val = in[POS_4D(n, x_in, y_in, c, IN_SHAPE)];

        if (x_in < 0 || x_in + kernelX > shapeParams[1] || y_in < 0 ||
            y_in + kernelY > shapeParams[2])
            val = max(val, 0.);

        for (int x_iter = max(x_in, 0);
             x_iter < min(x_in + kernelX, shapeParams[1]); ++x_iter)
        {
            for (int y_iter = max(y_in, 0);
                 y_iter < min(y_in + kernelY, shapeParams[2]); ++y_iter)
            {
                val = max(val, in[POS_4D(n, x_iter, y_iter, c, IN_SHAPE)]);
            }
        }

        out[POS_4D(n, x_out, y_out, c, OUT_SHAPE)] = val;
    }
}

template <PaddingType padding>
__global__ void pool2D_avg_nhwc_kernel(const float* in, float* out)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = n % shapeParams[7];
    n /= shapeParams[7];

    if (n < shapeParams[4] && x_out < shapeParams[5] &&
        y_out < shapeParams[6] && c < shapeParams[7])
    {
        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (kernelX - 1) / 2;
            y_in -= (kernelY - 1) / 2;
        }

        float val = 0.;
        for (int x_iter = max(x_in, 0);
             x_iter < min(x_in + kernelX, shapeParams[1]); ++x_iter)
        {
            for (int y_iter = max(y_in, 0);
                 y_iter < min(y_in + kernelY, shapeParams[2]); ++y_iter)
            {
                val += in[POS_4D(n, x_iter, y_iter, c, IN_SHAPE)];
            }
        }

        out[POS_4D(n, x_out, y_out, c, OUT_SHAPE)] = val / (kernelX * kernelY);
    }
}

template <PaddingType padding>
__global__ void pool2D_max_nchw_kernel(const float* in, float* out)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = n % shapeParams[5];
    n /= shapeParams[5];

    if (n < shapeParams[4] && c < shapeParams[5] && x_out < shapeParams[6] &&
        y_out < shapeParams[7])
    {
        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (kernelX - 1) / 2;
            y_in -= (kernelY - 1) / 2;
        }

        float val = 0.;
        if (x_in >= 0 && y_in >= 0)
            val = in[POS_4D(n, c, x_in, y_in, IN_SHAPE)];

        if (x_in < 0 || x_in + kernelX > shapeParams[2] || y_in < 0 ||
            y_in + kernelY > shapeParams[3])
            val = max(val, 0.);

        for (int x_iter = max(x_in, 0);
             x_iter < min(x_in + kernelX, shapeParams[2]); ++x_iter)
        {
            for (int y_iter = max(y_in, 0);
                 y_iter < min(y_in + kernelY, shapeParams[3]); ++y_iter)
            {
                val = max(val, in[POS_4D(n, c, x_iter, y_iter, IN_SHAPE)]);
            }
        }

        out[POS_4D(n,