#include "layerTests.h"

#ifdef CUDA_AVAILABLE
std::vector<MemoryLocation> LOCATIONS = {
    // clang-format off
    MemoryLocation::kHOST,
    MemoryLocation::kDEVICE
    // clang-format on
};
#else
std::vector<MemoryLocation> LOCATIONS = {
    // clang-format off
    MemoryLocation::kHOST
    // clang-format on
};
#endif

std::ostream& operator<<(std::ostream& os, MemoryLocation loc)
{
    if (loc == MemoryLocation::kHOST) return os << "HOST";
    return os << "DEVICE";
}

bool compareTensor(const RefTensor& refOutput, const HostTensor& output,
                   float eps, int tensorNum)
{
    EXPECT