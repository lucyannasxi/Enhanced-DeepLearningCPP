#ifndef TESTS_REF_TENSOR_H_
#define TESTS_REF_TENSOR_H_

#include "graphdl.h"
#include "randGen.h"
#include "tensorShape.h"

#include <initializer_list>
#include <ostream>
#include <vector>

using namespace graphdl;
using namespace graphdl::core;

class Coord
{
  public:
    Coord(const std::vector<int>& values);
    Coord(std::initializer_list<int> list);

    bool operator==(const Coord& other) const;
    Coord operator+(const Coord& c) const;

    unsigned size() const;

    int& operator[](size_t pos);
    const int& operator[](size_t pos) const;

    Coord cast(int start, int end) const;

  private:
    std::vector<int> mValues;
};

class Coord_iterator
{
  public:
    Coord_iterator(Coord c, Coord shape);
    Coord_iterator(const Coord_iterator& it) = default;
    Coord_iterator& operator=(const Coord_iterator& it) = default;

    Coord_iterator operator++();
    Coord_iterator operator++(int junk);

    bool operator==(const Coord_iterator& it) const;
    bool operator!=(const Coord_i