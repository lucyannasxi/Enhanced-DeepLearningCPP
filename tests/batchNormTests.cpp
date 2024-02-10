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
    return std::get<0>(std::get<0>(testCase));
}

int numAxes(const TestCase& testCase)
{
    return std::get<1>(std::get<0>(testCase));
}

UVec paramShape(const TestCase& testCase)
{
    UVec s = shape(testCase);
    int n = numAxes(testCase);

    UVec newShape;
    for (int i = n; i < s.size(); ++i) newShape.push_back(s[i]);

    return newShape;
}

MemoryLocation loc(const TestCase& testCase)
{
    return std::get<1>(testCase);
}

std::vector<std::tuple<UVec, int>> SHAPES = {
    // clang-format off
    {{1}, 1},
    {{10}, 1},
    {{10, 10}, 1},
    {{10, 10}, 2},
    {{2}, 1},
    {{2, 2, 2, 2}, 1},
    {{2, 2, 2, 2}, 2},
    {{2, 2, 2, 2}, 3},
    {{2, 2, 2, 2}, 4},
    {{4, 4, 4, 16}, 3},
    {{4, 2, 2, 32}, 3},
    {{4, 1, 1, 64}, 3},
    // clang-format on
};

class BatchNormTest : public LayerTest,
                      public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        setup(testCase);
        LayerBuilder builder = getBuilder(testCase);
        bool correct =
            runTest({mInput, mAlpha, mBeta}, {mOutput}, builder, 10e-4);
        EX