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
    if (!loss_out.IsInitialized() || loss_out.shape() != logits_in.shape()) {
      loss_out.AllocateTensor(logits_in.shape());
    }

    Tensor& logits_grad_out = context->output(0);
    if (!logits_grad_out.IsInitialized() || logits_grad_out.shape() != logits_in.shape()) {
      logits_grad_out.AllocateTensor(logits_in.shape());
    }
    
    LOG(DEBUG) << "sigmoid_logits_grad.name: " << context->outputs().at(0);

    auto labels = labels_in.matrix<T>();
    auto logits = logits_in.matrix<T>();
    auto loss = loss_out.matrix<T>();
    auto logits_grad = logits_grad_out.matrix<T>();

    auto d = context->template eigen_device<Device>();
    // sigmoid
    functor::SigmoidImpl<Device, T>::Compute(d, logits, logits_grad);
    // sigmoid cross entropy 
    functor::SigmoidCrossEntropyImpl<Device, T>::Compute(d, labels, logits_grad, loss); 
    // gradient of sigmoid cross entropy with logits
    logits_grad.device(d) = (logits_grad - labels);

    DLOG(INFO) << "after sigmoid_cross_entropy_with_logits. loss_out:\n" << loss
      << "\nlogits_grad_out:\n" << logits_grad;
  }
}; // class SigmoidCrossEntropyWithLogitsOp
OPENMI_REGISTER_OP_KERNEL_CPU(sigmoid_cross_entropy_with_logits, SigmoidCrossEntropyWithLogitsOp)

OPENMI_REGISTER_FILE_TAG(sigmoid_cross_entropy_with_logits);

}
