#include "unary_op.h"
#include "base/logging.h"
#include "op_registry.h"
using namespace openmi;

namespace openmi {

void UnaryOp::Initialize(OpKernelConstruction* context) {
  LOG(INFO) << "UnaryOp::Initialize ...";
  context_ = context;
}

void UnaryOp::Compute(OpKernelContext* context) {
  LOG(INFO) << "UnaryOp::Compute ...";
}

OPENMI_REGISTER_OP_KERNEL(UnaryOp, UnaryOp)
  .Device("CPU");

OPENMI_REGISTER_OP_KERNEL(Variable, UnaryOp)
  .Device("CPU");
OPENMI_REGISTER_OP_KERNEL(Placeholder, UnaryOp)
  .Device("CPU");
OPENMI_REGISTER_OP_KERNEL(MatMul, UnaryOp)
  .Device("CPU");
OPENMI_REGISTER_OP_KERNEL(Add, UnaryOp)
  .Device("CPU");
OPENMI_REGISTER_OP_KERNEL(Sigmoid, UnaryOp)
  .Device("CPU");

OPENMI_REGISTER_FILE_TAG(UnaryOp);

}
