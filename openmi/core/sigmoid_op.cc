#include "sigmoid_op.h"
#include "op_registry.h"

namespace openmi {

REGISTER_SIGMOID(float)
REGISTER_SIGMOID(double)
REGISTER_SIGMOID(int)

OPENMI_REGISTER_FILE_TAG(Sigmoid);

}
