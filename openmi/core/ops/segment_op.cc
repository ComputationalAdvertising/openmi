#include "local_device.h"
#include "segment_op.h"
#include "op_kernel.h"
#include "op_registry.h"
#include "tensor_utils.h"

using namespace openmi;

namespace openmi {

/*!
 * \brief Segment Operation
 */
template <typename Device, typename T, typename SegmentOpImpl> 
class SegmentOp : public OpKernel {
public: 
  void Initialize(OpKernelConstruction* ctx) override;
  
  void Compute(OpKernelContext* ctx) override;
private: 
  bool is_backward = false;
}; // class SegmentSumOp

template <typename Device, typename T, typename SegmentOpImpl> 
void SegmentOp<Device, T, SegmentOpImpl>::Initialize(OpKernelConstruction* ctx) {
  ctx->GetAttr<bool>("is_backward", &is_backward);
  DLOG(INFO) << "is_backward:" << is_backward;
}

template <typename Device, typename T, typename SegmentOpImpl> 
void SegmentOp<Device, T, SegmentOpImpl>::Compute(OpKernelContext* ctx) {
  auto& in0 = ctx->input(0);
  CHECK(in0.CheckTensorInitialized(ctx->inputs().at(0)));
  auto& in1 = ctx->input(1);
  CHECK(in1.CheckTensorInitialized(ctx->inputs().at(1)));

  auto& out = ctx->output();

  auto X = in0.matrix<T>();
  auto offset = in1.vec<int32_t>();

  DLOG(INFO) << "X: " << ctx->inputs().at(0)
             << ", shape: " << in0.shape().DebugString() << ", value:\n" << X;
  DLOG(INFO) << "offset: " << ctx->inputs().at(1) 
             << ", shape: " << in1.shape().DebugString() << ", value:\n" << offset;

  TensorShape expected_out_shape;
  int row_size = in1.shape().DimSize(0);
  int col_size = in0.shape().DimSize(1);
  if (is_backward) {
    row_size = offset(offset.dimension(0) - 1);
  }
  expected_out_shape.AddDim(row_size);
  expected_out_shape.AddDim(col_size);

  CheckAndAllocateTensor(ctx, expected_out_shape, out);

  DLOG(INFO) << "out shape:" << out.shape().DebugString();

  auto Y = out.matrix<T>();

  SegmentOpImpl::Compute(X, offset, Y);

  DLOG(INFO) << "Y:\n" << Y;
}

#define REGISTER_SEGMENT_OP(name, IMPL_TYPE, T) \
  OPENMI_REGISTER_OP_KERNEL(name, SegmentOp<CpuDevice, T, IMPL_TYPE<T, 2>>) \
    .Device("CPU") \
    .TypeConstraint<T>();

REGISTER_SEGMENT_OP(SegmentSum, SegmentSumOpImpl, float);
REGISTER_SEGMENT_OP(SegmentSumGrad, SegmentSumGradOpImpl, float);
REGISTER_SEGMENT_OP(SegmentMean, SegmentMeanOpImpl, float);
REGISTER_SEGMENT_OP(SegmentMeanGrad, SegmentMeanGradOpImpl, float);

OPENMI_REGISTER_FILE_TAG(segment_op);

} // namespace openmi