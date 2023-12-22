#ifndef GRAPHDL_CORE_LAYERS_ACTIVATION_H_
#define GRAPHDL_CORE_LAYERS_ACTIVATION_H_

#include "differentiableLayer.h"

#include <functional>

namespace graphdl
{
namespace core
{
namespace layers
{
enum class Activation
{
    kRELU = 0,
    kSIGMOID = 1,
    kTANH = 2,
    kSQUARE = 3,
    kABS = 4,
    kNEG = 5,
    kRECIPROCAL = 6,
    kLOG = 7,
    kSQRT = 8,
    kEXP = 9,
    kLEAKY_RELU = 10,
    kRELU_6 = 11,
    kELU = 12,
    kSOFTPLUS = 13,
    kSOFTSIGN = 14
};

class ActivationLayer : public DifferentiableLayer
{
  public:
    ActivationLayer(ID, const Tensor::SPtr&, Activation);

    TensorMap gradients(Tensor::SPtr, Tensor::SPtr) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    Activation mOp;
    std::function<float(float)>