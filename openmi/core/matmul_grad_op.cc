#include "matmul_grad_op.h"
#include "op_registry.h"
#include "base/register.h"

namespace openmi {

void ZeroslikeGradOp::Compute(OpKernelContext* context) {
  LOG(INFO) << "ZeroslikeGradOp::Compute ...";
}

void OneslikeGradOp::Compute(OpKernelContext* context) {
  LOG(INFO) << "OneslikeGradOp::Compute ...";
}

void MatMulGradOp::Compute(OpKernelContext* context) {
  LOG(INFO) << "MatMulGradOp::Compute ...";
  LOG(DEBUG) << "current node def:\n" << context->node_def().DebugString();
  // dX1 = dY * X2^T
  // dX2 = X1^ * dY
}

OPENMI_REGISTER_OP_KERNEL(MatMulGrad, MatMulGradOp)
  .Device("CPU");

OPENMI_REGISTER_OP_KERNEL(ZeroslikeGrad, ZeroslikeGradOp)
  .Device("CPU");

OPENMI_REGISTER_OP_KERNEL(OneslikeGrad, OneslikeGradOp)
  .Device("CPU");

//OPENMI_REGISTER_OP_KERNEL(AddGrad, AddGradOp).Device("CPU");

OPENMI_REGISTER_OP_KERNEL(VariableGrad, OneslikeGradOp)
  .Device("CPU");

OPENMI_REGISTER_OP_KERNEL(PlaceholderGrad, OneslikeGradOp)
  .Device("CPU");


OPENMI_REGISTER_FILE_TAG(matmul_grad_op);

}
