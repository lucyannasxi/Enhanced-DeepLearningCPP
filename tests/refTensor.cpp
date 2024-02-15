#include "refTensor.h"

#include <assert.h>

Coord::Coord(const std::vector<int>& values) : mValues(values) {}
Coord::Coord(std::initializer_list<int> list) : mValues(list) {}

bool Coord::operator==(const Coord& other) const
{
    if (mValues.size() != other.mValues.size()) return false;
    for (int i = 0; i < mValues.size(); ++i)
        if (mValues[i] != other.mValues[i]) return false;
    return true;
}

Coord Coord::operator+(const Coord& c) const
{
    assert(mValues.size() == c.mValues.size());

    std::vector<int> v = mValues;
    for (unsigned i = 0; i < mValues.size(); ++i) v[i] += c.mValues[i];

    return Coord(v);
}

unsigned Coord::size() const
{
    return mValues.size();
}

int& Coord::operator[](size_t pos)
{
    return mValues[pos];
}
const int& Coord::operator[](size_t pos) const
{
    return mValues[pos];
}

Coord Coord::cast(int start, int end) const
{
    std::vector<int> v(end - start, 0);
    for (int i = start; i < end; ++i) v[i - start] = mValues[i];
    return Coord(v);
}

Coord_iterator::Coord_iterator(Coord c, Coord shape) : mCoord(c), mShape(shape)
{
}

Coord_iterator Coord_iterator::operator++()
{
    Coord_iterator it = *this;

    unsigned p = mCoord.size() - 1;
    mCoord[p]++;
    while (p > 0)
    {
        if (mCoord[p] == mShape[p])
        {
            mCoord[p--] = 0;
            ++mCoord[p];
        }
        e