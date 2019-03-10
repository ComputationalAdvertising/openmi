#ifndef OPENMI_CORE_OPS_BINARY_ELEMENT_WISE_OP_H_ 
#define OPENMI_CORE_OPS_BINARY_ELEMENT_WISE_OP_H_ 

#include "numeric_op.h"

namespace openmi {

template <typename Device, typename T>
class AddOp : public BinaryElementWiseOp<T, AddOp<Device, T>> {
public:
  using BinaryElementWiseOp<T, AddOp<Device, T>>::BinaryElementWiseOp;

  template <int NDIMS>
  void Operate(OpKernelContext* context, Tensor& in0, Tensor& in1, Tensor& out) {
    Eigen::array<uint64_t, NDIMS> rdims;
    rdims[0] = 1;
    rdims[1] = in1.shape().DimSize(0);
    Eigen::array<uint64_t, NDIMS> bdims;
    bdims[0] = in1.shape().DimSize(0);
    bdims[1] = 1;
    auto d = context->template eigen_device<Device>(); 

    auto X0 = in0.flat<T>();
    auto X1 = in1.flat<T>();
    auto Y = out.flat<T>();
    Y.device(d) = X0 + X1.reshape(rdims).broadcast(bdims);

    LOG(INFO) << "name: " << context->name() << ", reshape(Y):\n" << out.tensor<T, NDIMS>() << ", " << out.shape().DebugString();
  }
}; // class AddOp

} // namespace openmi
#endif // OPENMI_CORE_OPS_BINARY_ELEMENT_WISE_OP_H_
