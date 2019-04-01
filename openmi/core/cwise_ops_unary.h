#ifndef OPENMI_CORE_OPS_CWISE_OPS_UNARY_H_
#define OPENMI_CORE_OPS_CWISE_OPS_UNARY_H_ 

#include "numeric_op.h"

namespace openmi {

/*!
 * \brief Input has the same shape as Output
 */
template <typename Device, typename FUNCTOR, typename T>
struct UnaryElementWiseOp : public UnaryOp<T, UnaryElementWiseOp<Device, FUNCTOR, T>> {
  template <int NDIMS>
  void Operate(OpKernelContext* context, Tensor& in, Tensor& out) {
    CHECK(in.shape().IsSameSize(out.shape())) << "shape not match. "
      <<"in:" << in.shape().DebugString() << ", out:" << out.shape().DebugString();
    typename FUNCTOR::func func;
    auto d = context->eigen_device<Device>();
    auto X = in.flat<T>();
    auto Y = out.flat<T>();
    Y.device(d) = X.unaryExpr(func);
    LOG(DEBUG) << "Y:\n" << Y;
  }
}; // struct UnaryElementWiseOp

} // namespace openmi

#define OPENMI_REGISTER_CWISE_UNARY_OP_UNIQ(name, functor, T) \
  OPENMI_REGISTER_OP_KERNEL(name, \
    ::openmi::UnaryOp<T, ::openmi::UnaryElementWiseOp<CpuDevice, functor<T>, T>>) \
  .Device("CPU") \
  .TypeConstraint<T>(); 

#define OPENMI_REGISTER_CWISE_UNARY_OP(name, functor) \
  OPENMI_REGISTER_CWISE_UNARY_OP_UNIQ(name, functor, float) \
  OPENMI_REGISTER_CWISE_UNARY_OP_UNIQ(name, functor, double) \
  OPENMI_REGISTER_CWISE_UNARY_OP_UNIQ(name, functor, int)

#endif // OPENMI_CORE_OPS_CWISE_OPS_UNARY_H_
