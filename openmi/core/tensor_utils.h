#ifndef OPENMI_CORE_FRAMEWORK_TENSOR_UTILS_H_
#define OPENMI_CORE_FRAMEWORK_TENSOR_UTILS_H_ 

#include "tensor.h"

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

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_TENSOR_UTILS_H_
