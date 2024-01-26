#ifndef GRAPHDL_CORE_TRAINERS_GRADIENT_DESCENT_H_
#define GRAPHDL_CORE_TRAINERS_GRADIENT_DESCENT_H_

#include "trainer.h"

namespace graphdl
{
namespace core
{
namespace trainers
{
class GradientDescentTrainer : public Trainer
{
  public:
    GradientDescentTrainer(float lr);

  private:
    Tensor::SPtr parseGradients(
        const Gradient