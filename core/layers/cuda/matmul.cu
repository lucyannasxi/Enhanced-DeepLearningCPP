#include "layers/matmul.h"

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
template <int TILE_SIZE, bool tran1, bool tran2>
__global__ void matmulKernel(int n, int m, int k, const float* X1,
                             const float* X2, float* Y)
{
    __shared__ float tile_X1[TILE_SIZE * TILE_SIZE];
    __shared__ float tile_X2[TILE_SIZE * TILE_SIZE];

    int pos = TILE_SIZE * threadIdx.x + threadIdx.y;
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    float tmp = 0.;

    for (int t = 0; t < m; t += TILE_SIZE)
    {
        if (t + threadIdx.y < m)
        {
            if (tran1)
                tile_X1[pos] = X1[n * (t + threadIdx.y) + row];
            else
                tile_X1[pos] = X1[m * row + t + threadIdx.y];
        }
        else
            tile_X1[pos] = 0.;

        if (t + threadIdx.x < m)
        {
            if (tran2)
                tile_X2[pos] = X2[m * col + t + threadIdx.x];
            else
                tile_X2[pos] = X2[k * (t + threadIdx.x) + col];
        }
        else
            tile_X2[pos] = 0.;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
           