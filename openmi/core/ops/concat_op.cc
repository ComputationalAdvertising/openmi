#include "core/framework/op_kernel.h"
#include "core/framework/op_registry.h"

namespace openmi {

template <typename Device, typename T, int NDIMS> 
void ConcatOpImpl(OpKernelContext* ctx) {
  Eigen::array<int, NDIMS> startIdx;
  Eigen::array<int, NDIMS> offset;

  auto& in0_shape = ctx->input(0).shape();
  for (int i = 0; i < in0_shape.Dims(); ++i) {
    startIdx[i] = 0;
    offset[i] = in0_shape.DimSize(i);
  }
  auto& out = ctx->output();
  auto Y = out.tensor<T, NDIMS>();

  Y.slice(startIdx, offset) = ctx->input(0).tensor<T, NDIMS>();

  for (int i = 1; i < ctx->inputs().size(); ++i) {
    auto& ini = ctx->input(i);
    startIdx[NDIMS - 1] = startIdx[NDIMS - 1] + ini.shape().DimSize(NDIMS - 1);
    offset[NDIMS - 1] = ini.shape().DimSize(NDIMS - 1);
    Y.slice(startIdx, offset) = ini.tensor<T, NDIMS>();
  } 
}

template <typename Device, typename T>
class ConcatOp : public OpKernel {
public:
  void Compute(OpKernelContext* ctx) override {
    auto& in0 = ctx->input(0);
    CHECK(in0.CheckTensorInitialized(ctx->inputs().at(0)));
    int rank = in0.shape().Dims();
    auto total_col_size = in0.shape().DimSize(rank - 1);
    DLOG(INFO) << "in0 shape: " << in0.shape().DebugString();

    TensorShape sub_shape(in0.shape());
    sub_shape.DeleteDim(rank - 1);

    for (int i = 1; i < ctx->inputs().size(); ++i) {
      auto& ini = ctx->input(i);
      CHECK(ini.CheckTensorInitialized(ctx->inputs().at(i)));
      auto ith_rank = ini.shape().Dims();
      CHECK(ith_rank = rank) << "input tensor rank not match.";
      TensorShape ini_sub_shape(ini.shape());
      ini_sub_shape.DeleteDim(rank - 1);
      CHECK(ini_sub_shape == sub_shape) << "[0, rank-1] shape not match.";
      total_col_size += ini.shape().DimSize(ith_rank - 1);

      DLOG(INFO) << "in" << i << " shape: " << ini.shape().DebugString();
    }

    TensorShape expected_out_shape(sub_shape);
    expected_out_shape.AddDim(total_col_size);

    DLOG(INFO) << "out shape: " << expected_out_shape.DebugString();

    auto& out = ctx->output();
    if (!out.IsInitialized() || out.shape() != expected_out_shape) {
      out.AllocateTensor(expected_out_shape);
    }

    switch (rank) {
#define NDIM_CASE(NDIMS) \
    case NDIMS: { \
      ConcatOpImpl<Device, T, NDIMS>(ctx); \
      break; \
    }
      NDIM_CASE(1);
      NDIM_CASE(2);
      NDIM_CASE(3);
      NDIM_CASE(4);
      NDIM_CASE(5);
      NDIM_CASE(6);
      NDIM_CASE(7);
      NDIM_CASE(8);
#undef NDIM_CASE

    default: 
      LOG(ERROR) << "opnemi only support handle up to Tensor::dims() up to 8. not " << rank;
      break;
    }
  }
}; // class ConcatOp  

OPENMI_REGISTER_OP_KERNEL_CPU(Concat, ConcatOp);

OPENMI_REGISTER_FILE_TAG(concat_op);

} // namespace openmi
