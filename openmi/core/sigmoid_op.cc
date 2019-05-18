#include "sigmoid_op_functor.h"
#include "cross_entropy_op_functor.h"
#include "op_kernel.h"
#include "op_registry.h"

namespace openmi {

template <typename Device, typename T>
class SigmoidOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) override {
    auto& logits_in = context->input(0);
    auto& sigmoid_out = context->output();

    if (!sigmoid_out.IsInitialized() || sigmoid_out.shape() != logits_in.shape()) {
      sigmoid_out.AllocateTensor(logits_in.shape());
    }

    auto logits = logits_in.matrix<T>();
    auto sigmoid = sigmoid_out.matrix<T>();

    auto d = context->template eigen_device<Device>();
    functor::SigmoidImpl<Device, T>::Compute(d, logits, sigmoid);

    LOG(DEBUG) << "sigmoid_op. sigmoid:\n" << sigmoid;
  }
}; // class SigmoidOp
OPENMI_REGISTER_OP_KERNEL_CPU(sigmoid, SigmoidOp)

template <typename Device, typename T>
class SigmoidCrossEntropyOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) override {
    auto& labels_in = context->input(0);
    auto& sigmoid_in = context->input(1);
    auto& loss_out = context->output();
    if (!loss_out.IsInitialized() || loss_out.shape() != sigmoid_in.shape()) {
      loss_out.AllocateTensor(sigmoid_in.shape());
    }

    auto labels = labels_in.matrix<T>();
    auto sigmoid = sigmoid_in.matrix<T>();
    auto loss = loss_out.matrix<T>();

    // sigmoid cross entropy
    auto d = context->template eigen_device<Device>();
    functor::SigmoidCrossEntropyImpl<Device, T>::Compute(d, labels, sigmoid, loss);

    LOG(DEBUG) << "after sigmoid cross entropy. loss:\n" << loss; 
  }
}; // class SigmoidCrossEntropyOp
OPENMI_REGISTER_OP_KERNEL_CPU(sigmoid_cross_entropy, SigmoidCrossEntropyOp)

OPENMI_REGISTER_FILE_TAG(sigmoid_op);

}
