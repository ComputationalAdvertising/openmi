#ifndef OPENMI_CORE_OPS_ONESLIKE_OP_H_
#define OPENMI_CORE_OPS_ONESLIKE_OP_H_

#include "op_kernel.h"

namespace openmi {

/*!
 * \brief (reversed) source node
 */ 
template <typename Device, typename T>
class OneslikeOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) override {
    Tensor& out = context->output();
    if (!out.IsInitialized()) {
      TensorShape out_shape("1");
      out.AllocateTensor(out_shape);
    }
    auto Y = out.flat<T>();
    Y.setConstant(static_cast<T>(1));
  }
}; // class OneslikeOp

} // namespace openmi 
#endif // OPENMI_CORE_OPS_ONESLIKE_OP_H_
