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
//!
using Shape = std::vector<unsigned int>;

//! \enum MemoryLocation
//! \brief Represents type of memory.
//!
enum class MemoryLocation
{
    ///< memory on host (CPU)
    kHOST = 0,
    ///< memory on device (GPU)
    kDEVICE = 1,
    ///< use memory on device if availble, host otherwise
    kDEVICE_IF_ENABLED = 2,
};

//! \brief Metatype for shared pointers.
//!
template <typename T>
using SharedPtr = std::shared_ptr<T>;

class ITensor;

//! \typedef ITensorPtr
//! \brief Shared pointer to ITensor.
//!
using ITensorPtr = SharedPtr<ITensor>;

class IGraph;

//! \typedef IGraphPtr
//! \brief Shared pointer to IGraph.
//!
using IGraphPtr = SharedPtr<IGraph>;

//! \class ITensor
//! \brief Interface representing tensor.
//!
class ITensor
{
  public:
    //! \fn getName
    //! \brief Returns name of the tensor.
    //!
    virtual std::string getName() const = 0;

    //! \fn setName
    //! \brief Sets new name for the tensor.
    //!
    virtual void setName(const std::string& name) = 0;

    //! \fn getShape
    //! \brief Returns shape of the tensor.
    //!
    virtual Shape getShape() const = 0;

    //! \fn eval
    //! \brief Evaulates tensor.
    //! \param inputs Map from names of input tensors to values.
    //!
    virtual HostTensor eval(const InputDict& inputs) = 0;

    virtual ~ITensor() {}
};

//! \class IGraph
//! \brief Interface representing computational graph.
//!
class IGraph
{
  public:
    //! \fn getName
    //! \brief Returns name of the graph.
    //!
    virtual std::string getName() const = 0;

    //! \fn setName
    //! \brief Sets new name for the graph.
    //!
    virtual void setName(const std::string& name) = 0;

    //! \fn getInputs
    //! \brief Returns map of all inputs to the graph.
    //!
    virtual std::map<std::string, ITensorPtr> getInputs() const = 0;

    //! \fn getWeights
    //! \brief Returns map of all weights in the graph.
    //!
    virtual std::map<std::string, ITensorPtr> getWeights() const = 0;

    virtual ~IGraph() {}
};

//! \class IInitializer
//! \brief Interface representing methods for initializing weights.
//!
class IInitializ