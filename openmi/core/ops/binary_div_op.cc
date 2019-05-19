#include "cwise_ops_binary.h"
#include "op_registry.h"

namespace openmi {

OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP(Div, DivFunctor)
OPENMI_REGISTER_FILE_TAG(binary_div_op);

}
