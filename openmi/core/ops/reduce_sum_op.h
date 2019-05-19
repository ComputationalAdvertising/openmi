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
    bool keep_dims = true;
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

    DLOG(INFO) << "X:\n" << X << "\tshape:" << in.shape().DebugString();
    
    auto d = context->template eigen_device<Device>();
    if (is_bcast) {
      Y.device(d) = X.broadcast(bcast_dims);
    } else {
      Y.device(d) = X;
    }
    
    DLOG(INFO) << "Y:\n" << Y << "\tshape: " << out.shape().DebugString();
  }
}; // class ReduceSumGradOp

// dims(in) >= dims(out)
template <typename Device, typename T>
class ReduceSumOp : public UnaryOp<T, ReduceSumOp<Device, T>> {
public:
  template <int NDIMS>
  void Operate(OpKernelContext* context, Tensor& in, Tensor& out) {
    bool keep_dims = true; 
    int axis = NDIMS - 1;  // default
    Eigen::array<int, 1> depth_dim;
    depth_dim[0] = axis;

    TensorShape expected_out_shape(in.shape());
    if (keep_dims) {
      expected_out_shape.SetDim(in.shape().Dims() - 1, 1);
    } else {
      expected_out_shape.DeleteDim(in.shape().Dims() - 1);
    }

    if (out.shape() != expected_out_shape) {
      out.ReallocateTensor(expected_out_shape);
    }

    Eigen::array<Eigen::DenseIndex, NDIMS> in_dims, out_dims;
    ReshapeTensor<NDIMS>(in, in_dims);
    ReshapeTensor<NDIMS>(out, out_dims);
    
    Eigen::TensorMap<Eigen::Tensor<T, NDIMS>> X(in.Base<T>(), in_dims);
    Eigen::TensorMap<Eigen::Tensor<T, NDIMS>> Y(out.Base<T>(), out_dims);

    DLOG(INFO) << "X:\n" << X << "\tshape:" << in.shape().DebugString();

    auto d = context->template eigen_device<Device>();
    Y.device(d) = X.sum(depth_dim).eval().reshape(out_dims);

    DLOG(INFO) << "Y:\n" << Y << "\tshape: " << out.shape().DebugString();
  }
}; // class ReduceSumOp

} // namespace openmi
#endif // OPENMI_CORE_OPS_REDUCE_SUM_OP_H_
