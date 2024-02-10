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
        EXPECT_TRUE(correct);
    }

    void testGradient(const TestCase& testCase)
    {
        setupGradient(testCase);
        LayerBuilder builder = getGradientBuilder(testCase);
        bool correct =
            runTest({mInput, mAlpha, mBeta, mOutputGrad},
                    {mInputGrad, mAlphaGrad, mBetaGrad}, builder, 10e-4);
        EXPECT_TRUE(correct);
    }

  private:
    void setup(const TestCase& testCase)
    {
        UniformGen gen(seed);
        mInput = RefTensor(shape(testCase), gen);
        mAlpha = RefTensor(paramShape(testCase), gen);
        mBeta = RefTensor(paramShape(testCase), gen);
        mOutput = RefTensor(shape(testCase));

        RefTensor mean(paramShape(testCase));
        RefTensor stddev(paramShape(testCase));

        calculateForward(testCase, mean, stddev);
    }

    void calculateForward(const TestCase& testCase, RefTensor& mean,
                          RefTensor& stddev)
    {
        UVec s = shape(testCase);
        int axes = numAxes(testCase);

        int cS = axes;
        int cE = s.size();

        int batchSize = 1;
        for (int i = 0; i < axes; ++i) batchSize *= s[i];

        Coord_iterator sBegin = shapeBegin(shape(testCase));
        Coord_iterator sEnd = shapeEnd(shape(testCase));

        for (Coord_iterator it = sBegin; it != sEnd; ++it)
            mean[it().cast(cS, cE)] += mInput[it()];
        for (int i = 0; i < mean.getCount(); ++i)
            mean.at(i) /= float(batchSize);
        for (Coord_iterator it = sBegin; it != sEnd; ++it)
        {
            Coord c = it().cast(cS, cE);
            stddev[c] += (mInput[it()] - mean[c]) * (mInput[it()] - mean[c]);
        }
        for (int i = 0; i < stddev.getCount(); ++i)
            stddev.at(i) /= float(batchSize);
        for (Coord_iterator it = sBegin; it != sEnd; ++it)
        {
            Coord c = it().cast(cS, cE);
            mOutput[it()] = mAlpha[c] * (mInput[it()] - mean[c]) /
                                std::sqrt(stddev[c] + EPS) +
                            mBeta[c];
        }
    }

    void setupGradient(const TestCase& testCase)
    {
        UniformGen gen(seed);
        mInput = RefTensor(shape(testCase), gen);
        mAlpha = RefTensor(paramShape(testCase), gen);
        mBeta = RefTensor(paramShape(testCase), gen);
        mOutputGrad = RefTensor(shape(testCase), gen);
        mOutput = RefTensor(shape(testCase));
        mInputGrad = RefTensor(shape(testCase));
        mAlphaGrad = RefTensor(paramShape(testCase));
        mBetaGrad = RefTensor(paramShape(testCase));

        RefTensor mean(paramShape(testCase));
        RefTensor stddev(paramShape(testCase));

        calculateForward(testCase, mean, stddev);

        UVec s = shape(testCase);
        int axes = numAxes(testCase);

        int cS = axes;
        int cE = s.size();

        int batchSize = 1;
        for (int i = 0; i < axes; ++i) batchSize *= s[i];

        Coord_iterator sBegin = shapeBegin(shape(testCase));
        Coord_iterator sEnd = shapeEnd(shape(testCase));

        for (Coord_iterator it = sBegin; it != sEnd; ++it)
            mBetaGrad[it().cast(cS, c