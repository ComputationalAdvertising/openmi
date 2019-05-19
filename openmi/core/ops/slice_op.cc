#include "numeric_op.h"

namespace openmi {

template <typename Device, typename T> 
class SliceOp : public UnaryOp<T, SliceOp<Device, T>> {
public: 
  void Initialize(OpKernelConstruction* context) override {
    context->GetAttr<int>("offset", &offset_);
    LOG(INFO) << "test_test offset " << offset_;
    LOG(INFO) << "test_test " << context->name();
  }

  template <int NDIMS> 
  void Operate(OpKernelContext* ctx, Tensor& in, Tensor& out) {
    LOG(INFO) << "in_name: " << ctx->inputs().at(0);
    LOG(INFO) << "out_name: " << ctx->name();
    // TODO get attr 'offset'
  }
private: 
  // The i-th input tensor
  int offset_ = 0;
}; // class SliceOp

OPENMI_REGISTER_UNARY_OP(Slice, SliceOp);

OPENMI_REGISTER_FILE_TAG(slice_op);

} // namespace openmi
