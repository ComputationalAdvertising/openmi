#include "oneslike_op.h"
#include "op_registry.h"
#include "base/register.h"

namespace openmi {

OPENMI_REGISTER_OP_KERNEL(Oneslike, OneslikeOp<CpuDevice, float>).Device("CPU").TypeConstraint<float>();
OPENMI_REGISTER_OP_KERNEL(Oneslike, OneslikeOp<CpuDevice, double>).Device("CPU").TypeConstraint<double>();

OPENMI_REGISTER_FILE_TAG(oneslike_op);

}
