#include "cwise_ops_unary.h"
#include "cwise_ops_unary_functor.h"
#include "op_registry.h"

namespace openmi {

template <typename T>
struct ReluOpImpl : BaseFunctor<T, ReluOpImpl<T>> {
  EIGEN_EMPTY_STRUCT_CTOR(ReluOpImpl)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE 
  const T operator()(const T& x) const {
    T zero = static_cast<T>(0);
    return x > zero ? x : zero;
  }
}; // struct ReluFunctor

template <typename T>
struct ReluOp : BaseFunctor<T, ReluOpImpl<T>> {
}; // struct ReluOp

template <typename T>
struct sigmoid_op : BaseFunctor<T, SigmoidFunctor<T>> {
};

OPENMI_REGISTER_CWISE_UNARY_OP(Sigmoid, sigmoid_op);

OPENMI_REGISTER_CWISE_UNARY_OP(Relu, ReluOpImpl)

OPENMI_REGISTER_FILE_TAG(cwise_ops_unary)

}
