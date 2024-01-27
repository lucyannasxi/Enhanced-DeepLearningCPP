//! \file graphdl.h
//! \brief Main header file of GraphDL library.
//!
//! \author Adam Jędrych adam.jedrych25@gmail.com
//!

#ifndef GRAPHDL_H_
#define GRAPHDL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

//! \namespace graphdl
//!
namespace graphdl
{
//! \typedef HostTensor
//! \brief Holds memory in host.
//! \details Used for providing data to graph
//!     and for receving outputs.
//!
using HostTensor = std::vector<float>;

//! \typedef InputDict
//! \brief Map from input name to HostTensor
//!
using InputDict = std::map<std::string, HostTensor>;

//! \typedef Shape
//! \brief Represents shape of the tensor.
/