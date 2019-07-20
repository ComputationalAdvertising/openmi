#include "assign_op.h"
#include "base/register.h"

namespace openmi {

OPENMI_REGISTER_UNARY_OP(Assign, AssignOp)
OPENMI_REGISTER_UNARY_OP(AssignGrad, AssignOp)
OPENMI_REGISTER_UNARY_OP(AddGrad, AssignOp)
OPENMI_REGISTER_UNARY_OP(MultiplyGrad, AssignOp)

OPENMI_REGISTER_FILE_TAG(register_ops);

} // namespace openmi