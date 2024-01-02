#include "layers/matmul.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
namespace
{
template <int TILE_SIZE, bool tran1, bool tran2>
__global__ void matmulKernel(int n, int m, int k, const