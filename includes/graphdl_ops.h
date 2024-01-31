//! \file graphdl_ops.h
//! \brief Header file with available operations.
//!
//! \author Adam Jędrych adam.jedrych25@gmail.com
//!
#ifndef GRAPHDL_OPS_H_
#define GRAPHDL_OPS_H_

#include "graphdl.h"

//! \namespace graphdl
namespace graphdl
{
//! \name Pointwise addition
//! \details If shapes of tensors don't match it tries to broadcast
//!     one to another. If one of the inputs is float then it
//!     considered as constant scalar.
///@{
//!
ITensorPtr add(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr add(float val, const ITensorPtr& t2);
ITensorPtr add(const ITensorPtr& t1, float val);
ITensorPtr operator+(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator+(float val, const ITensorPtr& t2);
ITensorPtr operator+(const ITensorPtr& t1, float val);
///@}

//! \name Pointwise substraction.
//! \details If shapes of tensors don't match it tries to broadcast
//!     one to another. If one of the inputs is float then it
//!     considered as constant scalar.
///@{
//!
ITensorPtr sub(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr sub(float val, const ITensorPtr& t2);
ITensorPtr sub(const ITensorPtr& t1, float val);
ITensorPtr operator-(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator-(float val, const ITensorPtr& t2);
ITensorPtr operator-(const ITensorPtr& t1, float val);
///@}

//! \name Pointwise multiplication.
//! \details If shapes of tensors don't match it tries to broadcast
//!     one to another. If one of the inputs is float then it
//!     considered as constant scalar.
///@{
//!
ITensorPtr mul(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr mul(float val, const ITensorPtr& t2);
ITensorPtr mul(const ITensorPtr& t1, float val);
ITensorPtr operator*(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator*(float val, const ITensorPtr& t2);
ITensorPtr operator*(const ITensorPtr& t1, float val);
///@}

//! \name Pointwise division.
//! \details If shapes of tensors don't match it tries to broadcast
//!     one to another. If one of the inputs is float then it
//!     considered as constant scalar.
///@{
//!
ITensorPtr div(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr div(float val, const ITensorPtr& t2);
ITensorPtr div(const ITensorPtr& t1, float val);
ITensorPtr operator/(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator/(float val, const ITensorPtr& t2);
ITensorPtr operator/(const ITensorPtr& t1, float val);
///@}

//! \fn ITensorPtr constant(float value, const Shape& shape,
//!                         MemoryLocation location)
//! \brief Creates constant tensor.
//!
ITensorPtr constant(float value, const Shape& shape, MemoryLocation location);

//! \fn ITensorPtr scalar(float value, MemoryLocation location)
//! \brief Creates constant scalar (tensor of dimension 0).
//!
ITensorPtr scalar(float value, MemoryLocation location);

//! \fn ITensorPtr matmul(const ITensorPtr& m1, const ITensorPtr& m2)
//! \brief Matrix multiplication.
//! Input tensors must be 