#include "no_gradient_op.h"
#include "op_registry.h"

namespace openmi {

OPENMI_REGISTER_OP_KERNEL_CPU(no_gradient, NoGradientOp);

OPENMI_REGISTER_FILE_TAG(no_gradient_op);

}
