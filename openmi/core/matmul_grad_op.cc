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
  // dX1 = dY * X2^T
  // dX2 = X1^ * dY
}

void AddGradOp::Compute(OpKernelContext* context) {
  LOG(INFO) << "AddGradOp::Compute ...";
}

void ReduceSumGradOp::Compute(OpKernelContext* context) {
  LOG(INFO) << "ReduceSumOp::Compute ...";
  auto& out = context->output();
  if (!out.IsInitialized()) {
    LOG(DEBUG) << context->name() << " is not initialized.";
  }
}

OPENMI_REGISTER_OP_KERNEL(MatMulGrad, MatMulGradOp)
  .Device("CPU");

OPENMI_REGISTER_OP_KERNEL(ZeroslikeGrad, ZeroslikeGradOp)
  .Device("CPU");

OPENMI_REGISTER_OP_KERNEL(OneslikeGrad, OneslikeGradOp)
  .Device("CPU");

OPENMI_REGISTER_OP_KERNEL(AddGrad, AddGradOp)
  .Device("CPU");

OPENMI_REGISTER_OP_KERNEL(ReduceSumGrad, ReduceSumGradOp)
  .Device("CPU");

OPENMI_REGISTER_OP_KERNEL(VariableGrad, OneslikeGradOp)
  .Device("CPU");

OPENMI_REGISTER_OP_KERNEL(PlaceholderGrad, OneslikeGradOp)
  .Device("CPU");


OPENMI_REGISTER_FILE_TAG(matmul_grad_op);

}
