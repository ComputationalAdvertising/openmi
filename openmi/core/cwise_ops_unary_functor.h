#ifndef OPENMI_CORE_OPS_CWISE_OPS_UNARY_FUNCTOR_H_
#define OPENMI_CORE_OPS_CWISE_OPS_UNARY_FUNCTOR_H_ 

#include "numeric_op.h"
#include <math.h>

namespace openmi {

template <typename T>
//struct SigmoidFunctor : BaseFunctor<T, SigmoidFunctor<T>> {
struct SigmoidFunctor {
  EIGEN_EMPTY_STRUCT_CTOR(SigmoidFunctor)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const T operator()(const T& x) const {
    auto one = static_cast<T>(1);
    auto norm_fn = [](T x, T eps=static_cast<T>(16)) -> T {
      return (x < -eps) ? -eps : ((x > eps) ? eps : x);
    };
    return one / (one + exp(- norm_fn(x)));
  }
}; // struct SigmoidFunctor

} // namespace openmi
#endif // OPENMI_CORE_OPS_CWISE_OPS_UNARY_FUNCTOR_H_
