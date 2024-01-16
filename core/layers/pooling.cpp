#include "pooling.h"

#include "abstractTensor.h"
#include "graph.h"
#include "graphdl.h"
#include "graphdl_ops.h"
#include "pooling_host.h"

#include <cassert>
#include <utility>

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
Tensor::SPtr createPoolingOutput(const Tensor::SPtr& t,
                                 const std::vector<int>& kernel,
                                 const std::vector<int>& strides,
                                 PaddingType padding, DataFormat dataFormat)
{
    TensorShape shape = t->getShape();

    if (dataFormat == DataFormat::kNHWC)
    {
        if (padding == PaddingType::kVALID)
        {
            shape[1] = ceil(shape[1] - kernel[0] + 1, strides[0]);
            shape[2] = ceil(shape[2] - kernel[1] + 1, strides[1]);
        }
        else  // padding == PaddingType::kSAME
        {
            shape[1] = ceil(shape[1], strides[0]);
            shape[2] = ceil(shape[2], strides[1]);
        }
    }
    else  // dataFormat == DataFormat::kNCHW
    {
        if (padding == PaddingType::kVALID)
        {
            shape[2] = ceil(shape[2] - kernel[0] + 1, strides[0]);
            shape[3] = ceil(shape[3] - kernel[1] + 1, strides[1]);
        }
        else  // padding == PaddingType::kSAME
        {
            shape[2] = ceil(shape[2], strides[0]);
            shape[3] = ceil(shape[3], strides[1]);
        }
    }

    return createTensor("", shape, t->getType());
}

void runPooling2DHost(const float* x, float* y, const std::vector<int>& inShape,
                      const std::vector<int>& outShape,
                      const std::vector<int>& kernel,
                      const std::vector<int>& strides, PoolingType pooling,
                      PaddingType padding, DataFormat dataFormat)
{
#define LAUNCH(format, pad)                                                    \
    {                                                                          \
        if (padding == PaddingType::kVALID)                                    \
            pool_##pad##_##format<PaddingType::kVALID>(                        \
                x, y, inShape, outShape, kernel, strides);                     \
        else                                                                   \
            pool_##pad##_##format<PaddingType::kSAME>(x, y, inShape, outShape, \
                                                      kernel, strides);        \
    }

    if (dataFormat == DataFormat::kNHWC)
    {
        if (pooling == PoolingType::kMAX)
            LAUNCH(nhwc, max)
        else  // pooling == PoolingType::kAVERAGE
            LAUNCH(nhwc, avg)
    }
    else  // dataFormat == DataFormat::kNCHW
    {
        if (pooling == PoolingType::kMAX)
            LAUNCH(nchw, max)
        else  // pooling == PoolingType::kAVERAGE
            LAUNCH(nchw, avg)
    }

#undef LAUNCH
}

void runPooling2DGradientHost(const float* x, const float* y, const float* yG,
                              float* xG, const std::vector<int>& inShape,
                              const std::vector<int>& outShape,
                              const std::vector<int>& kernel,
                              const std::vector<int>& strides,
                              PoolingType pooling, PaddingType padding,
                              DataFormat dataFormat)
{
#define LAUNCH(format, pad)                                        \
    {                                                              \
        if (padding == PaddingType::kVALID)                        \
            pool_grad_##pad##_##format<PaddingType::kVALID>(       \
                x, y, yG, xG, inShape, outShape, kernel, strides); \
        else                                                       \
            pool_grad_##pad##_##format<PaddingType::kSAME>(        \
                x, y, yG, xG, inShape, outShape, kernel, strides); \
    }

    if (dataFormat == DataFormat::kNHWC)
    {
        if (pooling == PoolingType::kMAX)
            LAUNCH(nhwc, max)
        else  // pooling == PoolingType::kAVERAGE
            LAUNCH(nhwc, avg)
    }
    else  // dataFormat == DataFormat::kNCHW
    {
        if (pooling == PoolingType::kMAX)
            LAUNCH(nchw, max)
        else  // pooling == PoolingType::kAVERAGE
            LAUNCH(nchw, avg)
    }

#undef LAUNCH
}

}  // namespace

PaddingType str2padding(const std::string& s)
{
    if (s == "SAME" || s == "same") return PaddingType::kSAME;
    if (s == "VALID" || s == "valid") return PaddingType::kVALID;

    throw std::runtime_error(
        R"(Wrong padding type, must be one of: "SAME", "VALID".)");
}

DataFormat str2format(const std::string& s)
{
    if (s == "NHWC" || s == "nhwc") return DataFormat::kNHWC;
    if (s == "NCHW" || s == "nchw") return DataFormat::kNCHW;

    throw std::runtime_error(
        R"(Wrong data format type, must be one of: "NHWC", "NCHW".)");
}

Pooling2DLayer::Pooling2DLayer(ID id, const Tensor::SPtr& t,
                               PoolingType pooling,
                               const std::vector<int>& kernel,
                               const std::vector<int>& strides,
                               PaddingType padding, DataFormat dataFormat)
    : DifferentiableLayer(
          id, {t},
          {createPoolingOutput(t, kernel, strides, padding, dataFormat)}),
      mPooling(pooling),
      mKernelWindow(kernel),
      mStrides(strides),
      mPadding(padding),
      mDataFormat(dataFormat),
      mGpuParams(MemoryType::kHOST_MEMORY, 13)
{
}

Layer::TensorMap Pooling2DLayer::gradients(Tensor::SPtr out,
                                           Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    Tensor::SPtr input = mInputs[0].lock();
    Layer::SPtr layer = createLayer<Pooling2DGradientLayer>(
        input, out, outGrad, mPooling, mKernelWindow, mStrides, mPadding,
        mDataFormat);
    return {{input, layer->getOutputs()[0]}};
}

void Pooling2DLayer::execute(const std::vector<float*>& inputs,
                             const std::vector<float*>& outputs,
                             const InputDict& /*inputDict*/)
{
    float* x = inputs[0];
    float* y = outputs[0];

    Tensor::SPtr tX = getInputs()[0];
    std::vector<int> inS