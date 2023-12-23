#include "batchNorm.h"

#include "abstractTensor.h"
#include "activation.h"
#include "elementwise.h"
#include "graph.h"
#include "graphdl_ops.h"
#include "reduce.h"

#include <cmath>

namespace graphdl
{
namespace 