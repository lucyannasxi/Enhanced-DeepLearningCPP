#include "assign.h"
#include "graphdl_ops.h"
#include "layerTests.h"

namespace
{
using TestCase = std::tuple<UVec, MemoryLocation>;
using ErrorTestCase = std::tuple<UVec, UVec>;

std::vector<UVec> SHAPES = {
    // clang-format off
    {},
    {1},
    {1, 1},
    {1, 1, 1},
    {1, 1, 1, 1},
    {1, 1, 1, 1, 1},
    {2},
    {2, 2},
    {2, 2, 2},
    {2, 2, 2, 2},
    {2, 2, 2, 2, 2}
    // clang-format on
};

std::vector<ErrorTestCase> ERROR_SHAPES = {
    // cla