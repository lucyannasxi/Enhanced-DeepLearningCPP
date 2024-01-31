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
//!