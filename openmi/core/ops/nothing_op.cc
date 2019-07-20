#include "nothing_op.h"
#include "op_registry.h"

namespace openmi {

OPENMI_REGISTER_OP_KERNEL_CPU(nothing, NothingOp);
OPENMI_REGISTER_OP_KERNEL_CPU(Placeholder, NothingOp);
OPENMI_REGISTER_OP_KERNEL_CPU(Variable, NothingOp);

OPENMI_REGISTER_FILE_TAG(nothing_op);

}
