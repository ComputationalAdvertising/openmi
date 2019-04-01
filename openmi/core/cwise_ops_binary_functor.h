#ifndef OPENMI_CORE_OPS_CWISE_OPS_BINARY_FUNCTOR_H_
#define OPENMI_CORE_OPS_CWISE_OPS_BINARY_FUNCTOR_H_ 

#include "numeric_op.h"
#include <math.h>
#include "base/logging.h"

namespace openmi {

template <typename T>
struct SigmoidGradFunctor {
  EIGEN_EMPTY_STRUCT_CTOR(SigmoidGradFunctor)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE 
  const T operator()(const T& y, const T& dy) const {
    return dy * y * (1 - y);
  }
}; // struct SigmoidGradFunctor

} // namespace openmi
#endif // OPENMI_CORE_OPS_CWISE_OPS_BINARY_FUNCTOR_H_
