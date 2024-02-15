#include "refTensor.h"

#include <assert.h>

Coord::Coord(const std::vector<int>& values) : mValues(values) {}
Coord::Coord(std::initializer_list<int> list) : mValues(list) {}

bool Coord::operator==(const Coord& other) const
{
    if (mValues.size() != other.mValues.size()) return false;
    for (int i = 0; i < mValues.size(); ++i)
     