#ifndef OPENMI_CORE_OPS_NOTHING_OP_H_
#define OPENMI_CORE_OPS_NOTHING_OP_H_ 

#include "op_kernel.h"

namespace openmi {

template <typename Device, typename T>
class NothingOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) override {
  }
}; // class NothingOp


} // namespace openmi
#endif // OPENMI_CORE_OPS_NOTHING_OP_H_