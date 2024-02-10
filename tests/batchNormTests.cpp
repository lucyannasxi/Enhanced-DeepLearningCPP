#include "abstractTensor.h"
#include "batchNorm.h"
#include "graphdl_ops.h"
#include "layerTests.h"

namespace
{
#define EPS 10e-8

using TestCase = std::tuple<std::tuple<UVec, int>, MemoryLocation>;

UVec shape(const TestCase& testCase)
{
    return std::get<0>(std