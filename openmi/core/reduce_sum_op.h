#ifndef OPENMI_CORE_OPS_REDUCE_SUM_OP_H_
#define OPENMI_CORE_OPS_REDUCE_SUM_OP_H_ 

#include "numeric_op.h"
#include "op_registry.h"

namespace openmi {

// dims(in) <= dims(out)
template <typename Device, typename T>
class ReduceSumGradOp : public UnaryOp<T, ReduceSumGradOp<Device, T>> {
public:
  template <int NDIMS>
  void Operate(OpKernelContext* context, Tensor& in, Tensor& out) {
    bool keep_dims = false;
    Eigen::array<Eigen::DenseIndex, NDIMS> in_dims, out_dims, bcast_dims;
    ReshapeTensor<NDIMS>(in, in_dims);
    ReshapeTensor<NDIMS>(out, out_dims);

    bool is_bcast = false;
    for (int i = 0; i < NDIMS; ++i) {
      bcast_dims[i] = in_dims[i] == 1 ? out_dims[i] : 1;
      if (bcast_dims[i] != 1) {
        is_bcast = true;
      }
    }
    
    Eigen::TensorMap<Eigen::Tensor<T, NDIMS>> X(in.Base<T>(), in_dims);
    Eigen::TensorMap<Eigen::Tensor<T, NDIMS>> Y(out.Base<T>(), out_dims);

    auto d = context->template eigen_device<Device>();
    if (is_bcast) {
      Y.device(d) = X.broadcast(bcast_dims);
    } else {
      Y.device(d) = X;
    }
    
    LOG(DEBUG) << "in: " << in.shape().DebugString() << ", value:\n" << X;
    LOG(DEBUG) << "out: " << out.shape().DebugString() << ", value:\n" << Y;
  }
}; // class ReduceSumGradOp

// dims(in) >= dims(out)
template <typename Device, typename T>
class ReduceSumOp : public UnaryOp<T, ReduceSumOp<Device, T>> {
public:
  template <int NDIMS>
  void Operate(OpKernelContext* context, Tensor& in, Tensor& out) {
    bool keep_dims = false; 
    int axis = NDIMS - 1;  // default
    Eigen::array<int, 1> depth_dim;
    depth_dim[0] = axis;

    if (in.shape().IsSameSize(out.shape())) {
      size_t out_rank_size = out.shape().Dims();
      if (!keep_dims) {
        out.shape().DeleteDim(out_rank_size - 1);
      } else {
        out.shape().SetDim(out_rank_size - 1, 1);
      }
      out.ReallocateTensor(out.shape());
    }

    Eigen::array<Eigen::DenseIndex, NDIMS> in_dims, out_dims;
    ReshapeTensor<NDIMS>(in, in_dims);
    ReshapeTensor<NDIMS>(out, out_dims);
    
    Eigen::TensorMap<Eigen::Tensor<T, NDIMS>> X(in.Base<T>(), in_dims);
    Eigen::TensorMap<Eigen::Tensor<T, NDIMS>> Y(out.Base<T>(), out_dims);

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
