#include "adagrad.h"

#include "abstractTrainer.h"
#include "graphdl_train.h"
#include "initializers/constantInitializer.h"
#include "layers/activation.h"
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
AdagradTrainer::AdagradTrainer(float lr, float eps)
    : mLearningRate(lr), mEpsilon(