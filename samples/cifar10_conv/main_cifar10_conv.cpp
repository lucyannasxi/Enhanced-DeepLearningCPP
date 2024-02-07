#include "graphdl.h"
#include "graphdl_ops.h"
#include "graphdl_train.h"
#include "readCIFAR10.h"
#include "utils.h"

#include <iostream>
#include <random>

const int BATCH_SIZE = 64;  // how many samples per computation
const int NUM_EPOCHS = 1;  // # of runs over whole dataset
const int PRINT_EVERY = 100;  // after how many batches print info
const float LEARNING_RATE = 0.001;  // learning parameter to the optimizer

#define Q(x) std::string(#x)
#define QUOTE(x) Q(x)

const std::vector<std::string> TRAIN_PATHS = {
    QUOTE(PROJECT_DIR) + "/samples/cifar10_conv/data_batch_1.bin",
    QUOTE(PROJECT_DIR) + "/samples/cifar10_conv/data_batch_2.bin",
    QUOTE(PROJECT_DIR) + "/samples/cifar10_conv/data_batch_3.bin",
    QUOTE(PROJECT_DIR) + "/samples/cifar10_conv/data_batch_4.bin",
    QUOTE(PROJECT_DIR) + "/samples/cifar10_conv/data_batch_5.bin",
};
const std::vector<std::string> VALID_PATHS = {
    QUOTE(PROJECT_DIR) + "/samples/cifar10_conv/test_batch.bin"};

#undef Q
#undef QUOTE

using namespace graphdl;

//! \fn conv2DAndMaxPool2D
//! \brief Helper function that does conv2D -> pool2D -> relu.
//!
ITensorPtr conv2DAndMaxPool2D(const ITensorPtr& x, const ITensorPtr& k)
{
    ITensorPtr a = conv2D(x, k, {1, 1}, "SAME", "NHWC");
    return relu(maxPool2D(a, {2, 2}, {2, 2}, "VALID", "NHWC"));
}

ComputationalGraph buildNetwork()
{
    MemoryLocation loc = MemoryLocation::kDEVICE_IF_ENABLED;

    // inputs
    ITensorPtr X = createInput("X", {BATCH_SIZE, 32, 32, 3}, loc);
    ITensorPtr Y = createInput("Y", {BATCH_SIZE, 10}, loc);

    ITensorPtr a = X;

    a = create_conv2D(a, 8, {3, 3}, {1, 1}, "SAME", "NHWC", "conv1");
    a = maxPool2D(a, {2, 2}, {2, 2}, "VALID", "NHWC");
    a = relu(a);

    a = create_conv2D(a, 16, {3, 3}, {1, 1}, "SAME", "NHWC", "conv2");
    a = maxPool2D(a, {2, 2}, {2, 2}, "VALID", "NHWC");
    a = relu(a);

    a = create_conv2D(a, 32, {3, 3}, {1, 1}, "SAME", "NHWC", "conv3");
    a = maxPool2D(a, {2, 2}, {2, 2}, "VALID", "NHWC");
    a = relu(a);

    a