
#include "abstractTensor.h"
#include "elementwise.h"
#include "graphdl_ops.h"
#include "layerTests.h"

namespace
{
using namespace graphdl::core::layers;

using TestCase =
    std::tuple<std::tuple<UVec, UVec>, Elementwise, MemoryLocation>;
using ErrorTestCase = std::tuple<std::tuple<UVec, UVec>, Elementwise>;

//!
//! n == 0 means to return first shape
//! n == 1 means to return second shape
//! n > 1  means to return the bigger shape
//!     (which is equal to the output shape)
//!
Shape shape(const TestCase& testCase, int n)
{
    if (n == 0)
        return std::get<0>(std::get<0>(testCase));
    else if (n == 1)
        return std::get<1>(std::get<0>(testCase));
    else
    {
        UVec v1 = shape(testCase, 0);
        UVec v2 = shape(testCase, 1);
        return v1.size() > v2.size() ? v1 : v2;
    }
}

Elementwise op(const TestCase& testCase)
{
    return std::get<1>(testCase);
}

MemoryLocation loc(const TestCase& testCase)
{
    return std::get<2>(testCase);
}

std::vector<std::tuple<UVec, UVec>> SHAPES = {
    // clang-format off
    {{}, {}},
    {{1}, {1}},
    {{1, 1}, {1, 1}},
    {{2}, {}},
    {{}, {2}},
    {{2}, {2}},
    {{2, 2}, {}},
    {{}, {2, 2}},
    {{2, 2}, {2}},
    {{2}, {2, 2}},
    {{2, 2}, {2, 2}},
    {{2, 2, 2}, {}},
    {{}, {2, 2, 2}},
    {{2, 2, 2}, {2}},
    {{2}, {2, 2, 2}},
    {{2, 2, 2}, {2, 2}},
    {{2, 2}, {2, 2, 2}},
    {{2, 2, 2}, {2, 2, 2}},
    {{2, 2, 2, 2}, {}},
    {{}, {2, 2, 2, 2}},
    {{2, 2, 2, 2}, {2, 2, 2, 2}},
    {{2, 2, 2, 2, 2}, {}},
    {{}, {2, 2, 2, 2, 2}},
    {{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}},
    {{2, 2, 2, 2, 2, 2}, {}},
    {{}, {2, 2, 2, 2, 2, 2}},
    {{2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2}},
    {{20, 20}, {}},
    /* {{}, {20, 20}}, */ // TODO: epsilon error
    {{20, 20}, {20}},
    {{20}, {20, 20}},
    {{20, 20}, {20, 20}},
    {{100, 100}, {}},
    /* {{}, {100, 100}}, */ // TODO: epsilon error
    {{100, 100}, {100, 100}},
    // clang-format on
};

std::vector<std::tuple<UVec, UVec>> ERROR_SHAPES = {
    // clang-format off
    {{2}, {5}},
    {{2, 3}, {2, 5}},
    {{3, 4}, {4, 3}},
    // clang-format on
};

std::vector<Elementwise> OPS = {
    // clang-format off
    Elementwise::kADD,
    Elementwise::kSUB,
    Elementwise::kMUL,
    Elementwise::kDIV
    // clang-format on
};

class ElementwiseTest : public LayerTest,
                        public testing::WithParamInterface<TestCase>
{
  public:
    void testBack(const TestCase& testCase)
    {
        setup(testCase, true);
        LayerBuilder builder = getBuilder(testCase, true);
        bool correct = runTest({mInput1, mInput2}, {mOutput}, builder);
        EXPECT_TRUE(correct);
    }

    void testFront(const TestCase& testCase)
    {
        setup(testCase, false);
        LayerBuilder builder = getBuilder(testCase, false);