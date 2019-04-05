#include "softmax_op_functor.h"
#include "op_kernel.h"
#include "op_registry.h"
#include "local_device.h"

namespace openmi {

// for error: class template partial specialization of 'SoftmaxFunctor' must occur at global scope
namespace functor {

template <typename Device, typename T>
struct SoftmaxFunctorBase {
  void operator()(const Device& d, typename TTypes<T>::Matrix logits, 
                  typename TTypes<T>::Matrix softmax, const bool is_log) {
    SoftmaxImpl<Device, T>::Compute(d, logits, softmax, is_log);
  }
}; // struct SoftmaxFunctorBase 

template <typename T>
struct SoftmaxFunctor<CpuDevice, T> : SoftmaxFunctorBase<CpuDevice, T> {};

/*
template <typename T>
struct SoftmaxFunctor<GpuDevcie, T> : SoftmaxFunctorBase<GpuDevice, T> {};
*/
} // namespace

/*!
 * \brief softmax operator. 
 *   input: m * n 
 *   output: m * k
 *   m: batch size 
 *   n: vector represent
 *   k: number of class (default = 2)
 */
template <typename Device, typename T>
class SoftmaxOp : public OpKernel {
public:
  void Initialize(OpKernelConstruction* ctx) override {
    ctx->GetAttr<bool>("is_log", &is_log_, ::openmi::AttrValue::kBool);
  }

  void Compute(OpKernelContext* context) override {
    Tensor& logits_in = context->input(0);
    CHECK(logits_in.shape().Dims() == 2) 
      << "softmax logits must be 2-dimensional";

    Tensor& softmax_out = context->output();
    if (!softmax_out.IsInitialized()) {
      softmax_out.AllocateTensor(logits_in.shape());
    }

    functor::SoftmaxFunctor<Device, T> functor;
    functor(context->template eigen_device<Device>(), 
        logits_in.matrix<T>(), softmax_out.matrix<T>(), is_log_);

    LOG(INFO) << "softmax out:\n" << softmax_out.matrix<T>();
  }
private:
  // apply to log softmax
  bool is_log_ = false;
}; // class SoftmaxOp

#define REGISTER_SOFTMAX_OP(T) \
  OPENMI_REGISTER_OP_KERNEL(Softmax, SoftmaxOp<CpuDevice, T>) \
    .Device("CPU").TypeConstraint<T>();

REGISTER_SOFTMAX_OP(float);
REGISTER_SOFTMAX_OP(double);
REGISTER_SOFTMAX_OP(int);

OPENMI_REGISTER_FILE_TAG(softmax_op);

} // namespace openmi
