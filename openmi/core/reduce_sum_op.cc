#include "reduce_sum_op.h"
#include "base/register.h"

namespace openmi {

OPENMI_REGISTER_UNARY_OP(ReduceSum, ReduceSumOp)
OPENMI_REGISTER_UNARY_OP(ReduceSumGrad, ReduceSumGradOp)

OPENMI_REGISTER_FILE_TAG(reduce_sum_op);

}
