
#include "abstractTensor.h"
#include "graphdl_ops.h"
#include "layerTests.h"
#include "matmul.h"

namespace
{
using namespace graphdl::core::layers;
using TestCase = std::tuple<std::pair<UVec, UVec>, MemoryLocation>;

std::vector<std::pair<UVec, UVec>> SHAPES = {
    // clang-format off
    {{1, 1}, {1, 1}},
    {{2, 2}, {2, 2}},
    {{2, 4}, {4, 2}},
    {{4, 2}, {2, 4}},
    {{2, 10}, {10, 3}},
    {{10, 10}, {10, 10}},
    {{1, 30}, {30, 1}},
    {{50, 50}, {50, 50}},
    {{10, 100}, {100, 20}}, // big test
    {{100, 100}, {100, 100}}
    // clang-format on
};

std::vector<std::pair<UVec, UVec>> ERROR_SHAPES = {
    // clang-format off
    {{1, 1}, {1}},
    {{2}, {2}},
    {{2, 2}, {2, 3, 3}},
    {{10, 1}, {2, 10}}
    // clang-format on
};

class MatmulTest : public LayerTest,
                   public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        setup(testCase);

        LayerBuilder builder = [&testCase](const HostVec& ins) {
            ITensorPtr input1 = createInput("i1", std::get<0>(testCase).first,
                                            std::get<1>(testCase));
            ITensorPtr input2 = createInput("i2", std::get<0>(testCase).second,
                                            std::get<1>(testCase));
            ITensorPtr output = matmul(input1, input2);
            initializeGraph();
            return HostVec({output->eval({{"i1", ins[0]}, {"i2", ins[1]}})});
        };
        bool correct = runTest({mInput1, mInput2}, {mOutput}, builder, 10e-5);
        EXPECT_TRUE(correct);
    }

    void testGradient(const TestCase& testCase)
    {
        setupGradient(testCase);

        unsigned n = std::get<0>(testCase).first[0];
        unsigned m = std::get<0>(testCase).first[1];
        unsigned k = std::get<0>(testCase).second[1];
        MemoryType type = memoryLocationToType(std::get<1>(testCase));
        LayerBuilder builder = [n, m, k, type](const HostVec& ins) {
            Tensor::SPtr i1 = core::getDefaultGraph()->addInput(
                "i1", createLayer<InputLayer>("i1", Shape({n, m}), type));
            Tensor::SPtr i2 = core::getDefaultGraph()->addInput(
                "i2", createLayer<InputLayer>("i2", Shape({m, k}), type));
            Tensor::SPtr outG = core::getDefaultGraph()->addInput(
                "outG", createLayer<InputLayer>("outG", Shape({n, k}), type));
            Tensor::SPtr out = matmul(i1, i2);
            Layer::SPtr layer =
                createLayer<MatmulGradientLayer>(i1, i2, out, outG);
            initializeGraph();

            std::vector<Tensor::SPtr> grads = layer->getOutputs();
            std::vector<ITensorPtr> igrads = {makeAbstractTensor(grads[0]),
                                              makeAbstractTensor(grads[1])};

            return eval(igrads,
                        {{"i1", ins[0]}, {"i2", ins[1]}, {"outG", ins[2]}});
        };
        bool correct = runTest({mInput1, mInput2, mOutputGrad},
                               {mGradient1, mGradient2}, builder, 10e-5);
        EXPECT_TRUE(correct);
    }

    void testWrongShapes(const TestCase& testCase)
    {
        ITensorPtr input1 = createInput("i1", std::get<0>(testCase).first,
                                        std::get<1>(testCase));
        ITensorPtr input2 = createInput("i2", std::get<0>(testCase).second,
                                        std::get<1>(testCase));
        ITensorPtr output;
        EXPECT_THROW({ output = matmul(input1, input2); }, std::runtime_error);
    }

  private:
    RefTensor mInput1, mInput2, mOutput, mOutputGrad, mGradient1, mGradient2;

    void setup(const TestCase& testCase)
    {
        UniformGen gen(seed);

        unsigned n = std::get<0>(testCase).first[0];
        unsigned m = std::get<0>(testCase).first[1];
        unsigned k = std::get<0>(testCase).second[1];

        mInput1 = RefTensor({int(n), int(m)}, gen);
        mInput2 = RefTensor({int(m), int(k)}, gen);
        mOutput = RefTensor({int(n), int(k)});

        // calculate reference output
        for (auto it = mOutput.begin(); it != mOutput.end(); ++it)
        {
            int x = it()[0], y = it()[1];
            mOutput[{x, y}] = 0.;
            for (int i = 0; i < int(m); ++i)
                mOutput[{x, y}] += mInput1[{x, i}] * mInput2[{i, y}];
        }
    }

    void setupGradient(const TestCase& testCase)
    {
        UniformGen gen(seed);

        unsigned n = std::get<0>(testCase).first[0];
        unsigned m = std::get<0>(testCase).first[1];
        unsigned k = std::get<0>(testCase).second[1];

        mInput1 = RefTensor({int(n), int(m)}, gen);
        mInput2 = RefTensor({int(m), int(k)}, gen);
        mOutputGrad = RefTensor({int(n), int(k)}, gen);
        mGradient1 = RefTensor({int(n), int(m)});
        mGradient2 = RefTensor({int(m), int(k)});

        // calculate reference gradient 1
        for (auto it = mInput1.begin(); it != mInput1.end(); ++it)
        {
            int x = it()[0], y = it()[1];
            mGradient1[{x, y}] = 0.;
            for (int i = 0; i < int(k); ++i)
                mGradient1[{x, y}] += mInput2[{y, i}] * mOutputGrad[{x, i}];
        }

        // calculate reference gradient 2
        for (auto it = mInput2.begin(); it != mInput2.end(); ++it)
        {
            int x = it()[0], y = it()[1];
            mGradient2[{x, y}] = 0.;
            for (int i = 0; i < int(n); ++i)
                mGradient2[{x, y}] += mInput1[{i, x}] * mOutputGrad[{i, y}];
        }
    }
};

TEST_P(MatmulTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TESTS(
    LayerTest, MatmulTest,
    Combine(ValuesIn(SHAPES), ValuesIn(LOCATIONS))
);

class MatmulErrorsTest : public MatmulTest
{
};
TEST_P(MatmulErrorsTest, testWrongShapes)
{
    testWrongShapes(GetParam());
}
INSTANTIATE_TESTS(
    LayerTest, MatmulErrorsTest,
    Combine(ValuesIn(ERROR_SHAPES), ValuesIn(LOCATIONS))
);

class MatmulGradientTest : public MatmulTest
{
};
TEST_P(MatmulGradientTest, testAPI)
{
    testGradient(GetParam());
}
INSTANTIATE_TESTS(
    LayerTest, MatmulGradientTest,
    Combine(ValuesIn(SHAPES), ValuesIn(LOCATIONS))
);

}  // namespace