#include "op_kernel.h"
#include "op_registry.h"
#include "sigmoid_op_functor.h"
#include "cross_entropy_op_functor.h"

namespace openmi {

template <typename Device, typename T>
class SigmoidCrossEntropyWithLogitsOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) override {
    const int kBatchDim = 0;
    const int kClassDim = 1;

    Tensor& labels_in = context->input(0);
    CHECK(labels_in.shape().Dims() == 2) 
      << "sigmoid_cross_entropy_with_logits [labels] must be 2-dimensional.";
    CHECK(labels_in.shape().DimSize(kClassDim) == 1)
      << "sigmoid num_class must be 1. actual:" 
      << labels_in.shape().DimSize(kClassDim);
      
    Tensor& logits_in = context->input(1);

    Tensor& loss_out = context->output();
    if (!loss_out.IsInitialized()) {
      loss_out.AllocateTensor(logits_in.shape());
    }

    auto labels = labels_in.matrix<T>();
    auto logits = logits_in.matrix<T>();
    auto loss = loss_out.matrix<T>();

    auto d = context->template eigen_device<Device>();
    // sigmoid
    functor::SigmoidImpl<Device, T>::Compute(d, logits, loss);
    // sigmoid cross entropy 
    functor::SigmoidCrossEntropyImpl<Device, T>::Compute(d, labels, loss, loss);

    LOG(DEBUG) << "after sigmoid_cross_entropy_with_logits. loss_out:\n" << loss;
  }
}; // class SigmoidCrossEntropyWithLogitsOp
OPENMI_REGISTER_OP_KERNEL_CPU(sigmoid_cross_entropy_with_logits, SigmoidCrossEntropyWithLogitsOp)

template <typename Device, typename T>
class SigmoidCrossEntropyWithLogitsGradOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) override {
    Tensor& labels_in = context->input(0);
    
    Tensor& logits_in = context->input(1);

    Tensor& logits_grad_out = context->output();
    if (!logits_grad_out.IsInitialized()) {
      logits_grad_out.AllocateTensor(logits_in.shape());
    }

    auto labels = labels_in.matrix<T>();
    auto logits = logits_in.matrix<T>();
    auto logits_grad = logits_grad_out.matrix<T>();

    auto d = context->template eigen_device<Device>();
    // sigmoid 
    functor::SigmoidImpl<Device, T>::Compute(d, logits, logits_grad);
    // gradient of sigmoid cross entropy with logits
    logits_grad.device(d) = (logits_grad - labels);

    LOG(DEBUG) << "gradient of sigmoid cross entropy:\n" << logits_grad_out.matrix<T>();
  }
}; // class SigmoidCrossEntropyWithLogitsGradOp

OPENMI_REGISTER_OP_KERNEL_CPU(sigmoid_cross_entropy_with_logits_grad, SigmoidCrossEntropyWithLogitsGradOp)

OPENMI_REGISTER_FILE_TAG(sigmoid_cross_entropy_with_logits);

}
