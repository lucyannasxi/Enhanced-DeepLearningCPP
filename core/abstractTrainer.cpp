#include "abstractTrainer.h"

#include "abstractTensor.h"

namespace graphdl
{
namespace core
{
AbstractTrainer::AbstractTrainer(Trainer::UPtr trainer)
    : mTrainer(std::move(trainer))
{
