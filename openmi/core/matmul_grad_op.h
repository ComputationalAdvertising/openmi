#ifndef OPENMI_CORE_OPS_MATMUL_OP_GRADIENT_H_
#define OPENMI_CORE_OPS_MATMUL_OP_GRADIENT_H_

#include "op_kernel.h"

namespace openmi {

class MatMulGradOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) override;
}; // class MatMulGradOp

class ZeroslikeGradOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) override;
}; // class ZeroslikeGradOp

class OneslikeGradOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) override;
}; // class OneslikeGradOp

} // namespace openmi
#endif // OPENMI_CORE_OPS_MATMUL_OP_GRADIENT_H_
