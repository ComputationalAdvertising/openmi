#ifndef OPENMI_CORE_OPS_SIGMOID_OP_H_
#define OPENMI_CORE_OPS_SIGMOID_OP_H_ 

#include "numeric_op.h"

namespace openmi {

template <typename Device, typename T>
class SigmoidOp : public UnaryOp<T> {
public:
  void Initialize(OpKernelConstruction* context) override;
  
  void Compute(OpKernelContext* context) override;

private:
  void Operate(OpKernelContext* context, Tensor& input, Tensor& output);

private:
  bool use_activation_func_ = false;
}; // class SigmoidOp

template <typename Device, typename T>
void SigmoidOp<Device, T>::Initialize(OpKernelConstruction* context) {
  // TODO use_activation_func_ 
  use_activation_func_ = false;
}

template <typename Device, typename T>
void SigmoidOp<Device, T>::Compute(OpKernelContext* context) {
  Tensor& input = context->input(0);
  Tensor& output = context->output();

  if (!output.IsInitialized()) {
    TensorShape shape;
    shape.AddDim(input.shape().DimSize(0));
    // AddDim(num_class);
    output.set_shape(shape);
    output.Init();
  }

  Operate(context, input, output);
}

template <typename Device, typename T>
void SigmoidOp<Device, T>::Operate(OpKernelContext* context, Tensor& input, Tensor& output) {
  auto X = input.tensor<T, 2>();
  auto Y = output.tensor<T, 2>();

  auto norm_fn = [](T x, T eps=static_cast<T>(16)) -> T {
    return (x < -eps) ? -eps : ((x > eps) ? eps : x);
  };

  auto d = context->template eigen_device<Device>();
    
  Eigen::array<int, 1> depth_dim({1});
  Y.device(d) = (-X.unaryExpr(norm_fn)).exp().sum(depth_dim);
  Y.device(d) = (Y + static_cast<T>(1)).inverse();
    
  LOG(DEBUG) << "SigmoidOp::Operate Y:\n" << Y.sum(depth_dim);
}

#define REGISTER_SIGMOID(T) \
    OPENMI_REGISTER_UNARY_OP(Sigmoid, openmi::SigmoidOp, T)

} // namespace openmi
#endif // OPENMI_CORE_OPS_SIGMOID_OP_H_ 
