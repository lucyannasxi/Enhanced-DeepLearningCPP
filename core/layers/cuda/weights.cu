#include "weights.h"

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

namespace graphdl
{
namespace core
{
namespace cuda
{
__global__ void setup_kernel(curandState* state)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, id, 0, &state[id]);
}

__global__ void initWeightsKernel(curandState* state, size_t size,
                                  float* output)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) output[id] = curand_uniform(state + id) * 2. - 1.;
}
