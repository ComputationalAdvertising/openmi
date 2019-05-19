#ifndef OPENMI_CORE_OPS_NO_GRADIENT_OP_H_
#define OPENMI_CORE_OPS_NO_GRADIENT_OP_H_ 

#include "op_kernel.h"

namespace openmi {

template <typename Device, typename T>
class NoGradientOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) override {
  }
}; // class NoGradientOp

} // namespace openmi
#endif // OPENMI_CORE_OPS_NO_GRADIENT_OP_H_
