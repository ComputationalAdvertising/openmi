#include "op_kernel.h"
#include "op_registry.h"

namespace openmi {

/*!
 * \brief <1, N>
 */
template <typename Device, typename T> 
class SliceOp : public OpKernel {
public: 
  void Initialize(OpKernelConstruction* context) override {
    context->GetAttr<int>("input_size", &input_size);
    DLOG(INFO) << "input_size: " << input_size;
  }

  void Compute(OpKernelContext* ctx) override {
    auto& in = ctx->input(0);
    CHECK(in.CheckTensorInitialized(ctx->inputs().at(0)));
    auto X = in.matrix<T>();

    size_t input_size = ctx->inputs().size();
    size_t output_size = ctx->outputs().size();
    CHECK(input_size - 1 == output_size) 
      << "size between input and output not match. "
      << "input size: " << input_size << ", output size: " << output_size 
      << ". cannot fetch output tensor shape according to input tensor name.";

    const int NDIMS = 2;

    Eigen::array<int, NDIMS> begin({{0, 0}});
    Eigen::array<int, NDIMS> size({{0, 0}});

    auto d = ctx->template eigen_device<Device>();

    for (size_t i = 0; i < ctx->outputs().size(); ++i) {
      auto& outi = ctx->output(i);
      Tensor* related_tensor = ctx->GetTensor(ctx->inputs().at(i+1));
      CHECK(related_tensor->IsInitialized()) 
        << __FUNCTION__ <<  " related tensor not initailized. "
        << "name: " << ctx->inputs().at(i+1);

      if (!outi.IsInitialized() || outi.shape() != related_tensor->shape()) {
        outi.AllocateTensor(related_tensor->shape());
      }

      auto Yi = outi.matrix<T>();
      
      for (size_t d = 0; d < outi.shape().Dims(); ++d) {
        if (NDIMS - 1 == d) {
          begin[d] = begin[d] + size[d];
        }
        size[d] = outi.shape().DimSize(d);
      }

      Yi.device(d) = X.slice(begin, size);

      DLOG(INFO) << "th-[" << i << "] out name: " << ctx->outputs().at(i) << ", shape: " << outi.shape().DebugString() << ", Yi:\n" << Yi;
    }
  }
private: 
  // 按照某一维度执行slice
  int input_size = 1;  
}; // class SliceOp

OPENMI_REGISTER_OP_KERNEL_CPU(Slice, SliceOp);

OPENMI_REGISTER_FILE_TAG(slice_op);

} // namespace openmi
