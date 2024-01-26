#include "momentum.h"

#include "abstractTrainer.h"
#include "gradientBuilder.h"
#include "graphdl_train.h"
#include "initializers/constantInitializer.h"
#include "layers/assign.h"
#include "layers/constant.h"
#include "layers/elementwise.h"
#include "layers/group.h"
#include "layers/queue.h"
#include "weights.h"
#include "weightsNamespaces.h"

namespace graphdl
{
namespace core
{
namespace trainers
{
MomentumTrainer::MomentumTrainer(float lr, float momentum)
    : mLearningRate(lr), mMomentum(momentum)
{
}

Tensor::SPtr MomentumTrainer::parseGradients(
    const GradientBuilder::TensorMap& grads) const
{
    GradientBuilder::TensorMap steps;
    for (const auto& grad : grads)
    {
        Tensor::SPtr w = grad.first;
        Tensor::SPtr wStep =
            weights("", w->getShape(), constantInitializer(0.), w->getType(),
                    core::TRAIN_WEIGHTS_NAMESPACE);
        steps.insert({w, wStep});
    }

