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
    std::vector<int> inShape = tX->getShape();
    std::vector<int> outShape = getOutputs()[0]->getShape();

    if (tX->getType() == MemoryType::kHOST_MEMORY)
        runPooling2DHost(x, y, inShape, outShape, mKernelWindow, mStrides,
                         mPooling, mPadding, mDataFormat);
#ifdef CUDA_AVAILABLE
    else  // inTensor->getType() == MemoryType::kDEVICE_MEMORY
        cuda::runPool2DDevice(x, y, mGpuParams.getValues(), mPooling, mPadding,
                              mDataFormat);
#endif
}

void Pooling2DLayer::initialize()
{
    std::vector<int> inShape = getInputs()[0]->getShape();
    std::vector<int> outShape = getOutputs()[0]->getShape();

    if (getInputs()[0]->getType() == MemoryType::kHOST_MEMORY)
    {
    }
#ifdef CUDA_AVAILABLE
    else
    {
        mGpuParams.allocate();
        int* values = mGpuParams.getValues();
        std::memcpy(values, inShape.data(), 4 * sizeof(int));
        std::memcpy(values + 4, outShape.data(), 4 * sizeof(int));
        std::memcpy(values + 8, mKernelWindow.data(), 2 * sizeof(int));
        std::memcpy(values + 10, mStrides.data(), 2 * sizeof(int));
    }
#endif
}

Pooling2DLayer::~Pooling2DLayer()
{
    mGpuParams.free();
}

Pooling2DGradientLayer::Pooling2DGradientLayer(
    ID id, const Tensor::SPtr& t, const Tensor::SPtr& out,
    const Tensor::SPtr& outGrad, PoolingType pooling, std::vector<int> kernel,
    std::vector<int> strides, PaddingType padding, DataFormat dataFormat)
    : Layer(id, {t, out, outGrad},
            {createTensor("", t->getShape(), t->getType())}),
      mPooling(pooling),
      mKernelWindow(std::move(kernel)),
      mStrides(std::move(strides)),
      mPadding(padding),
      mDataFormat(dataFormat),
      mGpuParams(MemoryType::kHOST_MEMORY, 13)
{
}

void Pooling2DGradientLayer::execute(const std::vector<float*>& inputs,
                                     const std::vector<float*>& outputs,
                                     const InputDict& /*inputDict*/)
{
    float* x = inputs[0];
    float* y = inputs[1];
    float* yGrad = inputs[2];
    float* xGrad = outputs[0];

    Tensor::SPtr tX = getInputs()[0];
    std::vector<int> inShape = tX->getShape();
    std::vector<int> outShape = getInputs()[1]->getShape();

    if (tX->getType() == MemoryType::kHOST_MEMORY)
        runPooling2DGradientHost(x, y, yGrad, xGrad, inShape, outShape,
                                 mKernelWindow, mStrides, mPooling, mPadding,
                                 mDataFormat);
#ifdef CUDA_AVAILABLE
    else  // outGradTensor->getType() == MemoryType::kDEVICE_MEMORY
        cuda::runPool2DGradientDevice(x, y, yGrad, xGrad,
                                      mGpuParams.getValues(), mPooling,
                                      mPadding, mDataFormat);
#endif
}

void Pooling2DGradientLayer::initialize()
{
    std::vector<int> inShape = getInputs()[0]->getShape();
    std::vector<int> outShape = getInputs()[1]->getShape();

    if (getInputs()[0]->getType() == MemoryType::kHOST_MEMORY)
    {
    }
#ifdef CUDA_AVAILABLE
    else
    {
        mGpuParams.allocate();
        int* values = mGpuParams.getValues();
        std::memcpy(values, inShape.data(), 4 * sizeof(int));
        std::memcpy(values + 4, outShape.data(), 4 * sizeof(int));
        std::memcpy(values + 8, mKernelWindow.data(), 2 * sizeof(int));
        std::memcpy(values + 10, mStrides.data(), 2 * sizeof(int));
    }
#endif
}

Pooling2DGradientLayer::~Pooling2DGradientLayer()
{
    mGpuParams.free();
}

}  // namespace layers

Tensor::SPtr pooling2D(const Tensor::SPtr& t, layers::PoolingType pooling,
                       const std::vector<int>& kernel,
                       const std::vector<int>& strides,
                       layers::PaddingType padding,
                       layers::DataFormat dataFormat)
{
    if (t->getShape().size() != 4)
        throw std::runtime_error("pool2D: wrong input shape");
    if (kernel.empty() || kernel.size() > 2)
        throw std::runtime_error("pool2D: wrong kernel");
    if (strides.empty() || strides.size() > 2)
        throw std::runtime_error("pool2D: wrong strides");

    for (int d : kernel)
        if (d <= 0)
            throw std::runtime_error("pool2D: kernel