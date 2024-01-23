#include "tensorShape.h"

#include <cassert>
#include <utility>

namespace graphdl
{
namespace core
{
TensorShape::TensorShape(Shape shape)
{
    mDims.reserve(mDims.size());
    for (int i : shape) mDims.push_back(int(i));
}

TensorShape::TensorShape(std::vector<int> vals) : mDims(std::move(vals))
{
    for (int i : mDims) assert(i >= 0);
}

TensorShape::TensorSh