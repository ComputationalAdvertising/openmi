#include "numeric_op.h"

namespace openmi {

/*!
 * \brief accmulate sum n op
 */
template <typename Device, typename T>
class AccumulateNOp : public OpKernel {
public:
  void Compute(OpKernelContext* ctx) override {
    std::vector<Tensor*> tensors_in;
    Tensor& in0 = ctx->input(0);
    tensors_in.push_back(&in0);
    auto shape0 = in0.shape();
    for (auto i = 1; i < ctx->inputs().size(); ++i) {
      Tensor& in_ith = ctx->input(i);
      CHECK(shape0.IsSameSize(in_ith.shape())) << " shape not match for AccmulateNOp inputs.";
      tensors_in.push_back(&in_ith);
    }

    Tensor& out = ctx->output();
    if (!out.IsInitialized()) {
      out.AllocateTensor(shape0);
    }

    auto d = ctx->template eigen_device<Device>();
    auto Y = out.flat<T>();
    Y = in0.flat<T>();
    for (int i = 1; i < ctx->inputs().size(); ++i) {
      LOG(DEBUG) << "in_i: " << i << ", value:\n" << ctx->input(i).flat<T>();
      Y.device(d) = Y + ctx->input(i).flat<T>();
    }
    LOG(DEBUG) << "accumulate_n_op Y:\n" << Y;

    /*
    functor::AccmulateNImpl<Device, T> impl;
    impl(context->template eigen_device<Device>(), tensors_in, out);
    */
  }
}; // struct AccumulateNOp

OPENMI_REGISTER_OP_KERNEL_CPU(accumulate_n, AccumulateNOp)
OPENMI_REGISTER_FILE_TAG(accumulate_n_op);

} // namespace openmi
