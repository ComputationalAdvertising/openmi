#ifndef OPENMI_CORE_OPS_REDUCE_SUM_OP_H_
#define OPENMI_CORE_OPS_REDUCE_SUM_OP_H_ 

#include "numeric_op.h"
#include "op_registry.h"

namespace openmi {

template <typename Device, typename T>
class ReduceSumOp : public UnaryOp<T, ReduceSumOp<Device, T>> {
public:
  template <int NDIMS>
  void Operate(OpKernelContext* context, Tensor& in, Tensor& out) {
    bool keep_dims = false;
    int ndims = in.shape().Dims();
    int axis = ndims - 1;
    Eigen::array<int, 1> depth_dim;
    depth_dim[0] = axis;
    
    std::vector<uint64_t> out_shape;
    out_shape.resize(keep_dims || ndims == 1 ? ndims : ndims - 1);
    for (auto i = 0; i < out_shape.size(); ++i) {
      out_shape[i] = context->input(0).shape().DimSize(i);
    }
    if (!out.IsInitialized() || out.shape().Dims() != out_shape.size()) {
      LOG(DEBUG) << "shape not match";
      TensorShape shape(out_shape);
      out.AllocateTensor(shape);
    }
    LOG(DEBUG) << "X: " << in.shape().DebugString();

    auto X = in.tensor<T, NDIMS>();
    auto Y = out.tensor<T, NDIMS>();
    
    auto d = context->template eigen_device<Device>();

    Y.device(d) = X.sum(depth_dim);

    LOG(DEBUG) << "Y: " << out.shape().DebugString() << "\nvalue:\n" << Y;
  }
}; // class ReduceSumOp

} // namespace openmi

#define OPENMI_REGISTER_REDUCE_SUM_OP_UNIQ(T) \
  OPENMI_REGISTER_OP_KERNEL(ReduceSum, ::openmi::UnaryOp<T, ::openmi::ReduceSumOp<CpuDevice, T>>) \
  .Device("CPU") \
  .TypeConstraint<T>(); 

#define OPENMI_REGISTER_REDUCE_SUM_OP() \
  OPENMI_REGISTER_REDUCE_SUM_OP_UNIQ(float) \
  OPENMI_REGISTER_REDUCE_SUM_OP_UNIQ(double) \
  OPENMI_REGISTER_REDUCE_SUM_OP_UNIQ(int)

#endif // OPENMI_CORE_OPS_REDUCE_SUM_OP_H_
