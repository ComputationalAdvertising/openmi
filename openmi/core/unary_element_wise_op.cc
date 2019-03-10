#include "unary_element_wise_op.h"
#include "op_registry.h"

namespace openmi {

OPENMI_REGISTER_UNARY_ELEMENT_WISE_OP(Relu, ReluOp)

OPENMI_REGISTER_FILE_TAG(UnaryElementWiseOp)

}
