#include "normalInitializer.h"

#ifdef CUDA_AVAILABLE
#include "layers/cuda/randomUtils.h"
#endif
#include "ab