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
    {{2, 2, 2, 2, 2}, 