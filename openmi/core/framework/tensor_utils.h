#ifndef OPENMI_CORE_FRAMEWORK_TENSOR_UTILS_H_
#define OPENMI_CORE_FRAMEWORK_TENSOR_UTILS_H_ 

#include "tensor.h"
#include "tensor_shape.h"
#include "op_kernel.h"
using namespace openmi;

namespace openmi {

template <int NDIMS>
inline void ReshapeTensor(Tensor& t, Eigen::array<Eigen::DenseIndex, NDIMS>& dims) {
  size_t rank_size = t.shape().Dims();
  for (size_t i = 0; i < rank_size; ++i) {
    dims[i] = t.shape().DimSize(i);
  }
  for (size_t i = rank_size; i < NDIMS; ++i) {
    dims[i] = 1;
  }
}

inline void CheckAndAllocateTensor(OpKernelContext* ctx, TensorShape& expected_out_shape, Tensor& tensor) {
  if (!tensor.IsInitialized() || tensor.shape() != expected_out_shape) {
    tensor.AllocateTensor(expected_out_shape);

    auto* related_tensor = ctx->GetTensor(ctx->related_node_name());
    if (related_tensor != nullptr && related_tensor->IsInitialized() && related_tensor->shape() != expected_out_shape) {
      related_tensor->AllocateTensor(expected_out_shape);
    }
  }
}

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_TENSOR_UTILS_H_
