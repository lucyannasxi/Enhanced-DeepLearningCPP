
#include "layerTests.h"

namespace
{
using TestCase = std::tuple<UVec, MemoryLocation>;

std::vector<UVec> SHAPES = {
    // clang-format off
    {1},
    {1, 1},
    {2},
    {2, 2},
    {2, 2, 2}
    // clang-format on
};

class InputTest : public LayerTest, public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        setup(testCase);

        LayerBuilder builder = [testCase](const HostVec& ins) {
            ITensorPtr input = createInput("input", std::get<0>(testCase),
                                           std::get<1>(testCase));
            initializeGraph();

            return HostVec({input->eval({{"input", ins[0]}})});
        };
        bool correct = runTest({mInput}, {mOutput}, builder);
        EXPECT_TRUE(correct);
    }

  private:
    RefTensor mInput, mOutput;

    void setup(const TestCase& testCase)
    {
        UniformGen gen(seed);

        mInput = RefTensor(std::get<0>(testCase), gen);
        mOutput = RefTensor(std::get<0>(testCase));

        // calculate reference output
        for (std::size_t i = 0; i < mInput.getCount(); ++i)
            mOutput.at(i) = mInput.at(i);
    }
};

TEST_P(InputTest, test)
{
    test(GetParam());
}
INSTANTIATE_TESTS(
    LayerTest, InputTest,
    Combine(ValuesIn(SHAPES), ValuesIn(LOCATIONS))
);
}  // namespace