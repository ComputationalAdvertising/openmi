#ifndef OPENMI_CORE_OPS_ASSIGN_OP_H_
#define OPENMI_CORE_OPS_ASSIGN_OP_H_ 

#include "numeric_op.h"

namespace openmi {

template <typename Device, typename T>
class AssignOp : public UnaryOp<T, AssignOp<Device, T>> {
public:
  template <int NDIMS>
  void ReduceRun(OpKernelContext* context, Tensor& in, Tensor& out) {
    LOG(DEBUG) << "ReduceRun name: " << context->name();
    Eigen::array<Eigen::DenseIndex, NDIMS> in_dims, out_dims;
    ReshapeTensor<NDIMS>(in, in_dims);
    ReshapeTensor<NDIMS>(out, out_dims);
    Eigen::TensorMap<Eigen::Tensor<T, NDIMS>> X(in.Base<T>(), in_dims);
    Eigen::TensorMap<Eigen::Tensor<T, NDIMS>> Y(out.Base<T>(), out_dims);
    Eigen::array<int, 1> depth_dim;
    int axis = NDIMS - 1;  // default

    if (in.shape().Dims() == out.shape().Dims()) {
      for (int i = 0; i < out.shape().Dims(); ++i) {
        if (out.shape().DimSize(i) == 1) {
          axis = i;
        }
      }
    }
    depth_dim[0] = axis;

    auto d = context->eigen_device<Device>();
    Y.device(d) = X.sum(depth_dim).eval().reshape(out_dims);

    LOG(DEBUG) << context->name() << ", Y:\n" << Y;
  }

  template <int NDIMS>
  void BroadcastRun(OpKernelContext* context, Tensor& in, Tensor& out) {
    LOG(DEBUG) << "BroadcastRun name: " << context->name();
    Eigen::array<Eigen::DenseIndex, NDIMS> in_dims, out_dims, bcast_dims;
    ReshapeTensor<NDIMS>(in, in_dims);
    ReshapeTensor<NDIMS>(out, out_dims);
    Eigen::TensorMap<Eigen::Tensor<T, NDIMS>> X(in.Base<T>(), in_dims);
    Eigen::TensorMap<Eigen::Tensor<T, NDIMS>> Y(out.Base<T>(), out_dims);
    
    for (int i = 0; i < NDIMS; ++i) {
      bcast_dims[i] = in_dims[i] == 1 ? out_dims[i] : 1;
    }

    auto d = context->eigen_device<Device>();
    Y.device(d) = X.broadcast(bcast_dims);
    
    LOG(DEBUG) << context->name() << ", Y:\n" << Y;
  }

  template <int NDIMS>
  void AssignEqualRun(OpKernelContext* context, Tensor& in, Tensor& out) {
    Eigen::array<Eigen::DenseIndex, NDIMS> in_dims, out_dims;
    ReshapeTensor<NDIMS>(in, in_dims);
    ReshapeTensor<NDIMS>(out, out_dims);
    Eigen::TensorMap<Eigen::Tensor<T, NDIMS>> X(in.Base<T>(), in_dims);
    Eigen::TensorMap<Eigen::Tensor<T, NDIMS>> Y(out.Base<T>(), out_dims);
    
    auto d = context->eigen_device<Device>();
    Y.device(d) = X;

    LOG(DEBUG) << context->name() << ", Y:\n" << Y;
  }

  template <int NDIMS>
  void Operate(OpKernelContext* context, Tensor& in, Tensor& out) {
    size_t in_rank_size = in.shape().Dims();
    size_t out_rank_size = out.shape().Dims();
    LOG(DEBUG) << "in: " << in.shape().DebugString() << ", out: " << out.shape().DebugString();
    if (in_rank_size > out_rank_size) {
      ReduceRun<NDIMS>(context, in, out);
    } else if (in_rank_size < out_rank_size) {
      BroadcastRun<NDIMS>(context, in, out);
    } else {
      bool is_reduce = false;
      for (size_t i = 0; i < in_rank_size; i++) {
        if (in.shape().DimSize(i) > out.shape().DimSize(i)) {
          is_reduce = true;
        }
      }
      if (is_reduce) {
        ReduceRun<NDIMS>(context, in, out);
      } else {
        AssignEqualRun<NDIMS>(context, in, out);
      }
    }
  }
}; // class AssignOp

} // namespace openmi
#endif // OPENMI_CORE_OPS_ASSIGN_OP_H_
