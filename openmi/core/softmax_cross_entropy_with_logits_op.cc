#include "softmax_cross_entropy_with_logits_op_functor.h"
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

    Tensor& loss_out = context->output();
    if (!loss_out.IsInitialized()) {
      TensorShape out_shape;
      uint64_t batch_size = labels_in.shape().DimSize(0);
      out_shape.AddDim(batch_size);
      out_shape.AddDim(1L);
      loss_out.AllocateTensor(out_shape);
    }

    functor::SoftmaxCrossEntropyWithLogitsFunctor<Device, T> functor;
    functor(context->template eigen_device<Device>(), labels_in.matrix<T>(), logits_in.matrix<T>(), loss_out.matrix<T>());
  }
};

#define OPENMI_REGISTER_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS(T) \
  OPENMI_REGISTER_OP_KERNEL(softmax_cross_entropy_with_logits,  \
                            SoftmaxCrossEntropyWithLogitsOp<CpuDevice, T>) \
    .Device("CPU").TypeConstraint<T>();

OPENMI_REGISTER_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS(float)
OPENMI_REGISTER_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS(double)
OPENMI_REGISTER_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS(int)

OPENMI_REGISTER_FILE_TAG(softmax_cross_entropy_with_logits);

}
