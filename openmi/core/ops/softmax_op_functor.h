#ifndef OPENMI_CORE_OPS_SOFTMAX_OP_FUNCTOR_H_
#define OPENMI_CORE_OPS_SOFTMAX_OP_FUNCTOR_H_ 

#include <Eigen/Core>
#include "tensor_types.h"
using namespace openmi;

namespace openmi {

namespace functor {
/*!
 * Functor used by SoftmaxOp to do the computation.
 */
template <typename Device, typename T>
struct SoftmaxFunctor {
  // logits: dim [batch_size, num_classes];
  // softmax: dim [batch_size, num_classes];
  void operator()(const Device& d, typename TTypes<T>::Matrix logits, typename TTypes<T>::Matrix softmax);
}; // struct SoftmaxFunctor

/*!
 * implementing SoftmaxFunctor::operator() or LogSoftmaxFunctor::operator()
 */
template <typename Device, typename T>
struct SoftmaxImpl {
  static void Compute(const Device& d, typename TTypes<T>::Matrix logits, typename TTypes<T>::Matrix softmax, const bool is_log) {
    const int kBatchDim = 0;
    const int kClassDim = 1;

    const int batch_size = logits.dimension(kBatchDim);
    const int num_classes = logits.dimension(kClassDim);

    Eigen::DSizes<int, 1> along_class(kClassDim);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);

    // shifted_logits = logits - max(logits along classes)
    auto shifted_logits = (logits - 
                           logits.maximum(along_class)
                            .eval()
                            .reshape(batch_by_one)
                            .broadcast(one_by_class));

    if (is_log) {
      // Calculate the log og the softmax 
      // softmax = \log { \frac {\exp(logits)} {\sum_{k=1}^K \exp(logits)} } 
      //         = logits - \log { \sum_{k=1}^K \exp(logits) }
      //         = shifted_logits - log(sum(exp(shifted_logits)))
      softmax.device(d) = shifted_logits;
      softmax.device(d) = (softmax - 
                           softmax.exp()
                            .sum(along_class)
                            .eval()
                            .reshape(batch_by_one)
                            .log()
                            .broadcast(one_by_class));
    } else {
      // softmax = \frac {\exp(shifted_logits)} {\sum_{k=1}^K \exp(shifted_logits)}
      softmax.device(d) = shifted_logits.exp();
      softmax.device(d) = (softmax * 
                           softmax.sum(along_class)
                            .inverse()
                            .eval()
                            .reshape(batch_by_one)
                            .broadcast(one_by_class));
    }
  }
}; // struct SoftmaxImpl




} // namespace functor
} // namespace openmi
#endif // OPENMI_CORE_OPS_SOFTMAX_OP_FUNCTOR_H_
