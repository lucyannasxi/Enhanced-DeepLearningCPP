#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

namespace graphdl
{
namespace core
{
namespace cuda
{
__global__ void setupKernel(curandState* state, size_t seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void uniformRandomKernel(curandState* state, float* memory,
                                    size_t size, float min, fl