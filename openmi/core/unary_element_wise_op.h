#ifndef OPENMI_CORE_OPS_UNARY_ELEMENT_WISE_OP_H_
#define OPENMI_CORE_OPS_UNARY_ELEMENT_WISE_OP_H_ 

#include "numeric_op.h"

namespace openmi {

template <typename Device, typename T>
class ReluOp : public UnaryElementWiseOp<T, ReluOp<Device, T>> {
public:
  using UnaryElementWiseOp<T, ReluOp<Device, T>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, Tensor& input, Tensor& output) {
    auto X = input.flat<T>();
    auto Y = output.flat<T>();

    auto d = context->eigen_device<Device>();

    Y.device(d) = X.cwiseMax(static_cast<T>(0));
  }
}; // class ReluOp 

} // namespace openmi
#endif // OPENMI_CORE_OPS_UNARY_ELEMENT_WISE_OP_H_
