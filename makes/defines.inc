# main paths
ROOT_DIR=$(shell dirname $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))
OUTDIR=$(ROOT_DIR)/build

GTEST_INCLUDE=/usr/local/include
GTEST_LIB_PATH=/usr/local/lib

# cuda settings

USE_CUDA ?= 0
CUDA_PATH ?= /usr/local/cuda

COMMON_CFLAGS=-std=c++14 -Wall -Wno-sign-compare -I$(ROOT_DIR)/includes
COMMON_LFLAGS=

ifeq ($(USE_CUDA),1)
    COM