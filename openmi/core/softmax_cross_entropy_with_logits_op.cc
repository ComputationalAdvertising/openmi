#include "cross_entropy_op_functor.h"
#include "op_kernel.h"
#include "op_registry.h"

namespace openmi {

namespace functor {

template <typename Device, typename T>
struct SoftmaxCrossEntropyWithLogitsFunctorBase {
  void operator()(const Device& d, typename TTypes<T>::Matrix labels, typename TTypes<T>::Matrix logits, typename TTypes<T>::Matrix loss) {
    SoftmaxCrossEntropyWithLogitsImpl<Device, T>::Compute(d, labels, logits, loss);
  }
}; // struct SoftmaxCrossEntropyWithLogitsFunctorBase 

template <typename T>
struct SoftmaxCrossEntropyWithLogitsFunctor<CpuDevice, T> : SoftmaxCrossEntropyWithLogitsFunctorBase<CpuDevice, T> {};

} // namespace functor

template <typename Device, typename T>
class SoftmaxCrossEntropyWithLogitsOp : public OpKernel {
public:
  void Initialize(OpKernelConstruction* ctx) override {
  }

  void Compute(OpKernelContext* context) override {
    Tensor& labels_in = context->input(0);
    CHECK(labels_in.shape().Dims() == 2) 
      << "softmax_cross_entropy_with_logits [labels] must be 2-dimensional.";

    Tensor& logits_in = context->input(1);
    CHECK(logits_in.shape().Dims() == 2) 
      << "softmax_cross_entropy_with_logits [logits] must be 2-dimensional.";
    
    CHECK(labels_in.shape().IsSameSize(logits_in.shape()))
      << "shape of labels and logits not match";
    
    TensorShape expected_loss_out_shape;
    expected_loss_out_shape.AddDim(labels_in.shape().DimSize(0));
    expected_loss_out_shape.AddDim(1L);

    Tensor& loss_out = context->output();
    if (!loss_out.IsInitialized() || loss_out.shape() != expected_loss_out_shape) {
      loss_out.AllocateTensor(expected_loss_out_shape);
    }

    functor::SoftmaxCrossEntropyWithLogitsFunctor<Device, T> functor;
    functor(context->template eigen_device<Device>(), 
            labels_in.matrix<T>(), 
            logits_in.matrix<T>(), 
            loss_out.matrix<T>());
  }
};

template <typename Device, typename T>
class SoftmaxCrossEntropyWithLogitsGradOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) override {
    Tensor& labels_in = context->input(0);
    CHECK(labels_in.shape().Dims() == 2) 
      << "softmax_cross_entropy_with_logits [labels] must be 2-dimensional.";

    Tensor& logits_in = context->input(1);
    CHECK(logits_in.shape().Dims() == 2) 
      << "softmax_cross_entropy_with_logits_grad [logits] must be 2-dimensional.";

    CHECK(labels_in.shape().IsSameSize(logits_in.shape()))
      << "shape of labels and logits not match";
    
    // gradient of cross entropy with logits
    Tensor& logits_grad_out = context->output();
    if (!logits_grad_out.IsInitialized() || logits_grad_out.shape() != logits_in.shape()) {
      logits_grad_out.AllocateTensor(logits_in.shape());
    }

    auto labels = labels_in.matrix<T>();
    auto logits = logits_in.matrix<T>();
    auto logits_grad = logits_grad_out.matrix<T>();

    auto d = context->template eigen_device<Device>();
    // softmax 
    functor::SoftmaxImpl<Device, T>::Compute(d, logits, logits_grad, false);
    // dX = d(logits) = dY * (Prob - Label)
    logits_grad.device(d) = (logits_grad - labels);
    LOG(DEBUG) << "gradient of softmax cross entropy:\n" << logits_grad_out.matrix<T>();
  }
}; // class SoftmaxCrossEntropyWithLogitsGradOp

OPENMI_REGISTER_OP_KERNEL_CPU(softmax_cross_entropy_with_logits, SoftmaxCrossEntropyWithLogitsOp)
OPENMI_REGISTER_OP_KERNEL_CPU(softmax_cross_entropy_with_logits_grad, SoftmaxCrossEntropyWithLogitsGradOp)

OPENMI_REGISTER_FILE_TAG(softmax_cross_entropy_with_logits);

}
