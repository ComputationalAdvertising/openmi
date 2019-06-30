#ifndef OPENMI_CORE_OPS_EMBEDDING_LOOKUP_OP_H_
#define OPENMI_CORE_OPS_EMBEDDING_LOOKUP_OP_H_

#include "op_kernel.h"
#include "op_registry.h"

namespace openmi {

template <typename Deivce, typename T>
class EmbeddingLookupOp : public OpKernel {
public: 
  void Initialize(OpKernelConstruction* ctx) override;
  
  void Compute(OpKernelContext* ctx) override;

private: 
  bool x_always_one = false;
  bool offset_always_one = false;
}; // class EmbeddingLookupOp

template <typename Device, typename T> 
void EmbeddingLookupOp<Device, T>::Initialize(OpKernelConstruction* ctx) {
  // TODO 获取参数
  // check input_size != 3
}

template <typename Device, typename T> 
void EmbeddingLookupOp<Device, T>::Compute(OpKernelContext* ctx) {
  auto& in0 = ctx->input(0);
  CHECK(in0.CheckTensorInitialized(ctx->inputs().at(0)));
  auto& in1 = ctx->input(1);
  CHECK(in1.CheckTensorInitialized(ctx->inputs().at(1)));
  auto& in2 = ctx->input(2);
  CHECK(in2.CheckTensorInitialized(ctx->inputs().at(2)));

  auto& out = ctx->output();

  auto W = in0.matrix<T>();
  auto X = in1.matrix<T>();
  auto offset = in2.vec<int>();

  TensorShape expected_out_shape;
  // length(offset) * embedding_size
  expected_out_shape.AddDim(in2.shape().DimSize(0));  // batch_size
  expected_out_shape.AddDim(in0.shape().DimSize(1));  // embedding_size

  if (!out.IsInitialized() || out.shape() != expected_out_shape) {
    out.AllocateTensor(expected_out_shape);

    auto* related_node = ctx->GetTensor(ctx->related_node_name());
    if (related_node != nullptr && 
        related_node->IsInitialized() && 
        related_node->shape() != expected_out_shape) {
      related_node->AllocateTensor(expected_out_shape);
    }
  }

  auto d = ctx->template eigen_device<Device>();
  auto Y = out.matrix<T>();

  // compute device
  if (x_always_one) {
    Y.device(d) = W;
  } else {
    // 矩阵相乘
  }

  if (!offset_always_one) {
    // step2: Segment Sum, 调用SegmentSum Functor方法；
    // step3: 是否需要pooling
  }
  
  DLOG(INFO) << "name:" << ctx->name() << ", Y:\n" << Y;
} // method Compute

OPENMI_REGISTER_OP_KERNEL_CPU(embedding_lookup, EmbeddingLookupOp);

OPENMI_REGISTER_FILE_TAG(embedding_lookup);

} // namespace openmi
#endif // OPENMI_CORE_OPS_EMBEDDING_LOOKUP_OP_H_