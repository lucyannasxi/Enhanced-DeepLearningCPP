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
    