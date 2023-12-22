
#include "activation.h"

#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"

#include <cassert>
#include <cmath>
#include <utility>

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
template <Activation act>
float activation(float x);

template <>
float activation<Activation::kRELU>(float x)
{
    return x >= 0. ? x : 0.;
}
template <>
float activation<Activation::kSIGMOID>(float x)
{
    return 1. / (1. + std::exp(-x));
}
template <>
float activation<Activation::kTANH>(float x)
{
    return std::tanh(x);
}
template <>
float activation<Activation::kSQUARE>(float x)
{
    return x * x;
}
template <>
float activation<Activation::kABS>(float x)
{
    return std::abs(x);
}
template <>
float activation<Activation::kNEG>(float x)
{
    return -x;
}
template <>
float activation<Activation::kRECIPROCAL>(float x)
{
    return 1. / x;
}
template <>
float activation<Activation::kLOG>(float x)
{
    return std::log(x);
}
template <>
float activation<Activation::kSQRT>(float x)
{
    return std::sqrt(x);
}
template <>
float activation<Activation::kEXP>(float x)
{
    return std::exp(x);
}
template <>
float activation<Activation::kLEAKY_RELU>(float x)
{
    return x >= 0. ? x : 0.01 * x;
}
template <>
float activation<Activation::kRELU_6>(float x)
{
    return x >= 0. ? (x <= 6. ? x : 6.) : 0.;
}
template <>
float activation<Activation::kELU>(float x)
{
    return x >= 0. ? x : std::exp(x) - 1.;
}
template <>
float activation<Activation::kSOFTPLUS>(float x)
{
    return std::log(std::exp(x) + 1.);
}
template <>
float activation<Activation::kSOFTSIGN>(float x)
{
    return x / (std::abs(x) + 1.);
}

template <Activation act>
void activationHost(const float* x, float* y, size_t size)
{
    for (std::size_t i = 0; i < size; ++i) y[i] = activation<act>(x[i]);
}

template <Activation act>
float activationGradient(float x, float o);
template <>
float activationGradient<Activation::kRELU>(float x, float /* o */)
{
    return x >= 0. ? 1. : 0.;
}
template <>
float activationGradient<Activation::kSIGMOID>(float /* x */, float o)
{
    return o * (1. - o);
}
template <>
float activationGradient<Activation::kTANH>(float /* x */, float o)
{
    return 1. - o * o;
}
template <>
float activationGradient<Activation::kSQUARE>(float x, float /* o */)
{
    return 2. * x;
}
template <>
float activationGradient<Activation::kABS>(float x, float /* o */)
{
    return x >= 0. ? 1. : -1;
}
template <>
float activationGradient<Activation::kNEG>(float /* x */, float /* o */)
{
    return -1;
}
template <>
float activationGradient<Activation::kRECIPROCAL>(float /* x */, float o)
{
    return -1. * o * o;
}
template <>
float activationGradient<Activation::kLOG>(float x, float /* o */)
{
    return 1. / x;
}
template <>
float activationGradient<Activation::kSQRT>(float /* x */, float o)
{
    return 1. / (2 * o);
}
template <>
float activationGradient<Activation::kEXP>(float /* x */, float o)
{
    return o;
}
template <>
float activationGradient<Activation::kLEAKY_RELU>(float x, float /* o */)
{
    return x >= 0. ? 1. : 0.01;
}
template <>
float activationGradient<Activation::kRELU_6>(float x, float /* o */)
{
    return x >= 0. ? (x <= 6. ? 1. : 0.) : 0.;
}
template <>
float activationGradient<Activation::kELU>(float x, float o)
{
    return x >= 0. ? 1. : o + 1.;