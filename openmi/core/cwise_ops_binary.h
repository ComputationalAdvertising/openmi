#ifndef OPENMI_CORE_OPS_CWISE_OPS_BINARY_H_ 
#define OPENMI_CORE_OPS_CWISE_OPS_BINARY_H_ 

#include "cwise_ops_binary_functor.h"
#include "tensor_types.h"
#include "numeric_op.h"

namespace openmi {

extern void UpdateOneVectorReshape(Tensor& t, uint64_t* reshape, int dim_size);
extern void UpdateMultiDimReshape(Tensor& t, uint64_t* reshape, int dim_size); 
extern void ReshapeTensor(Tensor& t, uint64_t* reshape, int dims);

template <typename Device, typename FUNCTOR, typename T, int NDIMS>
inline void BinaryOperate(OpKernelContext* context, Tensor& in0, Tensor& in1, Tensor& out) {

}

template <typename Device, typename FUNCTOR, typename T>
struct BinaryElementWiseOp : public BinaryOp<T, BinaryElementWiseOp<Device, FUNCTOR, T>> {
  template <int NDIMS>
  void Operate(OpKernelContext* context, Tensor& in0, Tensor& in1, Tensor& out) {

  uint64_t lreshape[NDIMS], rreshape[NDIMS], lbcast[NDIMS], rbcast[NDIMS];

  const bool in0_vec = in0.IsVector();
  const bool in1_vec = in1.IsVector();

  if (in0_vec && in1_vec) {
    lreshape[0] = in0.shape().DimSize(0);
    rreshape[0] = in1.shape().DimSize(0);
  } else if (in0_vec && !in1_vec) {
    UpdateOneVectorReshape(in0, lreshape, NDIMS);
    UpdateMultiDimReshape(in1, rreshape, NDIMS);
  } else if (!in0_vec && in1_vec) {
    UpdateMultiDimReshape(in0, lreshape, NDIMS);
    UpdateOneVectorReshape(in1, rreshape, NDIMS);
  } else {
    UpdateMultiDimReshape(in0, lreshape, NDIMS);
    UpdateMultiDimReshape(in1, rreshape, NDIMS);
  }

  LOG(INFO) << "lreshape: " << lreshape[0];
  LOG(INFO) << "rreshape: " << rreshape[0];

  TensorShape out_shape;
  bool is_lbcast = false, is_rbcast = false;
  for (int i = 0; i < NDIMS; ++i) {
    lbcast[i] = lreshape[i] == 1 ? rreshape[i] : 1;
    if (lbcast[i] != 1) {
      is_lbcast = true;
    }
    rbcast[i] = rreshape[i] == 1 ? lreshape[i] : 1;
    if (rbcast[i] != 1) {
      is_rbcast = true;
    }
    out_shape.AddDim(lreshape[i] != 1 ? lreshape[i] : rreshape[i]);

    if (lreshape[i] != 1 && rreshape[i] != 1 && lreshape[i] != rreshape[i]) {
      LOG(INFO) << "left: " << in0.shape().DebugString();
      LOG(INFO) << "right: " << in1.shape().DebugString();
      std::runtime_error("Dim error.");
    }
  }
  
  LOG(INFO) << "out_shape: " << out_shape.DebugString();

  if (!out.IsInitialized()) {
    out.AllocateTensor(out_shape);
  }

  auto X0 = in0.tensor<T, NDIMS>();
  auto X1 = in1.tensor<T, NDIMS>();
  LOG(INFO) << "X0:\n" << X0; 
  LOG(INFO) << "X1:\n" << X1; 
  auto Y = out.tensor<T, NDIMS>();

  Eigen::array<Eigen::DenseIndex, NDIMS> lreshape_dims, rreshape_dims, lbcast_dims, rbcast_dims;
  for (int i = 0; i < NDIMS; ++i) {
    lreshape_dims[i] = lreshape[i];
    rreshape_dims[i] = rreshape[i];
    lbcast_dims[i] = lbcast[i];
    rbcast_dims[i] = rbcast[i];
  }

  //LOG(INFO) << "is_lbcast: " << is_lbcast << ", is_rbcast: " << is_rbcast; 
  
  typename FUNCTOR::func func;
  auto d = context->eigen_device<Device>();
  if (is_lbcast && is_rbcast) {
    Y.device(d) = X0.reshape(lreshape_dims).broadcast(lbcast_dims).binaryExpr(
      X1.reshape(rreshape_dims).broadcast(rbcast_dims), func);
  } else if (is_lbcast && !is_rbcast) {
    Y.device(d) = X0.reshape(lreshape_dims).broadcast(lbcast_dims).binaryExpr(
      X1.reshape(rreshape_dims), func);
  } else if (!is_lbcast && is_rbcast) {
    Y.device(d) = X0.reshape(lreshape_dims).binaryExpr(
      X1.reshape(rreshape_dims).broadcast(rbcast_dims), func);
  } else {
    Y.device(d) = X0.reshape(lreshape_dims).binaryExpr(
      X1.reshape(rreshape_dims), func);
  }
  /*
    auto X00 = X0.reshape(lreshape_dims);
    if (is_lbcast) {
      X00 = X00.broadcast(lbcast_dims);
    }
    auto X11 = X1.reshape(rreshape_dims);
    if (is_rbcast) {
      X11 = X11.broadcast(rbcast_dims);
    }
    Y.device(d) = X00.binaryExpr(X11, typename FUNCTOR::func());
  */

  LOG(INFO) << "Y:\n" << Y;
}

}; // struct BinaryElementWiseOp

// 'a+b'
template <typename T>
struct AddFunctor : BaseFunctor<T, Eigen::internal::scalar_sum_op<T>> {
  static const bool use_bcast_optimization = true;
};
// 'a-b'
template <typename T>
struct SubFunctor : BaseFunctor<T, Eigen::internal::scalar_difference_op<T>> {
  static const bool use_bcast_optimization = true;
};
// 'a*b'
template <typename T>
struct MulFunctor : BaseFunctor<T, Eigen::internal::scalar_product_op<T>> {};
// 'a/b'
template <typename T>
struct DivFunctor : BaseFunctor<T, Eigen::internal::scalar_quotient_op<T>> {};
// '2^'
//template <typename T>
//struct PowFunctor : BaseFunctor<T, Eigen::internal::scalar_binary_pow_op_google<T, T>> {};
// 'a > b ? a : b'
template <typename T>
struct MaxFunctor : BaseFunctor<T, Eigen::internal::scalar_max_op<T>> {};
// 'a < b ? a : b'
template <typename T>
struct MinFunctor : BaseFunctor<T, Eigen::internal::scalar_min_op<T>> {};

template <typename T>
struct AddFunctorCustomOp {
  EIGEN_EMPTY_STRUCT_CTOR(AddFunctorCustomOp)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const T operator()(const T& x1, const T& x2) const {
    return x1 + x2;
  }
};

template <typename T>
struct AddFunctorCustom : BaseFunctor<T, AddFunctorCustomOp<T>> {
};

template <typename T>
struct SigmoidGradOp : BaseFunctor<T, SigmoidGradFunctor<T>> {
};

} // namespace openmi 

#define OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP_TYPE(name, functor, T) \
  OPENMI_REGISTER_OP_KERNEL(name,  \
      ::openmi::BinaryOp<T, openmi::BinaryElementWiseOp<CpuDevice, functor<T>, T>>) \
    .Device("CPU") \
    .TypeConstraint<T>();

#define OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP(name, functor) \
  OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP_TYPE(name, functor, float) \
  OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP_TYPE(name, functor, double) \
  OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP_TYPE(name, functor, int)

#endif // OPENMI_CORE_OPS_CWISE_OPS_BINARY_H_
