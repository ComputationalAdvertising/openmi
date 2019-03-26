#include "matmul_grad_op.h"
#include "gradient_op_registry.h"
#include "base/register.h"

namespace openmi {

void ZeroslikeGradOp::Compute(OpKernelContext* context) {
  LOG(INFO) << "ZeroslikeGradOp::Compute ...";
}

void OneslikeGradOp::Compute(OpKernelContext* context) {
  LOG(INFO) << "OneslikeGradOp::Compute ...";
}

void SigmoidGradOp::Compute(OpKernelContext* context) {
  LOG(INFO) << "SigmoidGradOp::Compute ...";
  // dX = y * (1 - y) * dY
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

OPENMI_REGISTER_GRADIENT_OP_KERNEL(MatMul, MatMulGradOp)
  .Device("CPU");

OPENMI_REGISTER_GRADIENT_OP_KERNEL(Zeroslike, ZeroslikeGradOp)
  .Device("CPU");

OPENMI_REGISTER_GRADIENT_OP_KERNEL(Oneslike, OneslikeGradOp)
  .Device("CPU");

OPENMI_REGISTER_GRADIENT_OP_KERNEL(Add, AddGradOp)
  .Device("CPU");

OPENMI_REGISTER_GRADIENT_OP_KERNEL(ReduceSumGrad, ReduceSumGradOp)
  .Device("CPU");

OPENMI_REGISTER_GRADIENT_OP_KERNEL(Sigmoid, SigmoidGradOp)
  .Device("CPU");

OPENMI_REGISTER_GRADIENT_OP_KERNEL(Variable, OneslikeGradOp)
  .Device("CPU");

OPENMI_REGISTER_GRADIENT_OP_KERNEL(Placeholder, OneslikeGradOp)
  .Device("CPU");


OPENMI_REGISTER_FILE_TAG(matmul_grad_op);

}
