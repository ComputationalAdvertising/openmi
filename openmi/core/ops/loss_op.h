#ifndef OPENMI_CORE_OPS_LOSS_OP_H_
#define OPENMI_CORE_OPS_LOSS_OP_H_ 

#include "op_kernel.h"

namespace openmi {

template <typename T>
class LossOp : public OpKernel {
public:
  void Initialize(OpKernelConstruction* context) override {
  }

}; // class LossOp

template <typename T, typename CHILD>
class LossOpImpl : public LossOp<T> {
public:
  void Compute(OpKernelContext* context) override {
    // TODO 
  }
}; // class LossOpImpl

} // namespace openmi
#endif // OPENMI_CORE_OPS_LOSS_OP_H_ 
