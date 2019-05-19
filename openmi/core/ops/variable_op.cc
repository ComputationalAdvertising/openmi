#include "variable_op.h"
#include "op_registry.h"

namespace openmi {

OPENMI_REGISTER_OP_KERNEL(Variable, VariableOp)
  .Device("CPU");

OPENMI_REGISTER_OP_KERNEL(Placeholder, VariableOp)
  .Device("CPU");

OPENMI_REGISTER_FILE_TAG(Variable)

}
