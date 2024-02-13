#include "abstractTensor.h"
#include "graphdl_ops.h"
#include "layerTests.h"
#include "reduce.h"
#include "utils.h"

namespace
{
using namespace graphdl::core::layers;
using Param = std::tuple<UVec, int>;
using TestCase = std::tuple<Param, ReduceType, MemoryLocation>;

std::vector<Param> PARAMS = {
    // clang-format off
    {{1}, 1},
    {{1, 1}, 1},
    {{1, 1}, 2},
    {{1, 1, 1}, 1},
    {{1, 1, 1}, 2},
    {{1, 1, 1}, 3},
    {{2}, 1},
    {{2, 2}, 1},
    {{2, 2}, 2},
    {{2, 2, 2}, 1},
    {{2, 2, 2}, 2},
    {{2, 2, 2}, 3},
    {{2, 2, 2, 2}, 1},
    {{2, 2, 2, 2}, 2},
    {{2, 2, 2, 2}, 3},
    {{2, 2, 2, 2}, 4},
    {{2, 2, 2, 2, 2}, 1},
    {{2, 2, 2, 2, 2}, 2},
    {{2, 2, 2, 2, 2}, 3},
    {{2, 2, 2, 2, 2}, 4},
    {{2, 2, 2, 2, 2}, 5},
    {{2, 2, 2, 2, 2, 2}, 1},
    {{2, 2, 2, 2, 2, 2}, 2},
    {{2, 2, 2, 2, 2, 2}, 3},
    {{2, 2, 2, 2, 2, 2}, 4},
    {{2, 2, 2, 2, 2, 2}, 5},
    {{2, 2, 2, 2, 2, 2}, 6},
    {{10}, 1},
    {{10, 10}, 1},
    {{10, 10}, 2},
    {{10, 10, 10}, 1},
    {{10, 10, 10}, 2},
    {{10, 10, 10}, 3},
    {{2, 100}, 1},
    {{2, 100}, 2},
    {{100, 100}, 1},
    {{100, 100}, 2},
    {{100, 100, 10}, 1}, // big test for multiple reductions
    {{100, 100, 10}, 2} // big test for multiple reductions
    // clang-format on
};

std::vector<ReduceType> REDUCE_TYPES = {
    // clang-format off
    ReduceType::kSUM,
    ReduceType::kMAX,
    ReduceType::kMIN
    // clang-format on
};

UVec inputShape(const TestCase& testCase)
{
    return std::get<0>(std::get<0>(testCase));
}

int numAxes(const TestCase& testCase)
{
    return std::get<1>(std::get<0>(testCase));
}

ReduceType reduceType(const TestCase& testCase)
{
    return std::get<1>(testCase);
}

Memory