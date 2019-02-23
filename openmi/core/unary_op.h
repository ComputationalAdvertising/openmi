#ifndef OPENMI_CORE_OPS_UNARY_OP_H_ 
#define OPENMI_CORE_OPS_UNARY_OP_H_ 

#include "op_kernel.h"

namespace openmi {

class UnaryOp : public OpKernel {
public:
  void Initialize(OpKernelConstruction* context) override;

  void Compute(OpKernelContext* context) override;
}; // class UnaryOp

} // namespace openmi
#endif // OPENMI_CORE_OPS_UNARY_OP_H_
