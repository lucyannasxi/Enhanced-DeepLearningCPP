
#include "graphdl.h"
#include "graphdl_ops.h"
#include "graphdl_train.h"
#include "readMNIST.h"
#include "utils.h"

#include <iostream>
#include <random>

// learning parameters
const int BATCH_SIZE = 64;  // how many samples per computation
const int NUM_EPOCHS = 1;  // # of runs over whole dataset
const int PRINT_EVERY = 100;  // after how many batches print info
const float LEARNING_RATE = 0.1;  // learning parameter to the optimizer

#define Q(x) std::string(#x)
#define QUOTE(x) Q(x)

const std::string TRAIN_IMAGES_PATH =
    QUOTE(PROJECT_DIR) + "/samples/mnist/train-images-idx3-ubyte";
const std::string TRAIN_LABELS_PATH =
    QUOTE(PROJECT_DIR) + "/samples/mnist/train-labels-idx1-ubyte";
const std::string VALID_IMAGES_PATH =
    QUOTE(PROJECT_DIR) + "/samples/mnist/t10k-images-idx3-ubyte";
const std::string VALID_LABELS_PATH =
    QUOTE(PROJECT_DIR) + "/samples/mnist/t10k-labels-idx1-ubyte";

#undef Q
#undef QUOTE

using namespace graphdl;

//! \fn buildNetwork
//! \brief Builds computation graph.
//!
ComputationalGraph buildNetwork()
{
    MemoryLocation loc = MemoryLocation::kDEVICE_IF_ENABLED;

    IInitializerPtr init = uniformInitializer(-1., 1., 0);