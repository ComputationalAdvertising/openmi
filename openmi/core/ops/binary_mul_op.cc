#include "cwise_ops_binary.h"
#include "op_registry.h"

namespace openmi {

OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP(Mul, MulFunctor)
OPENMI_REGISTER_FILE_TAG(binary_mul_op);

}
