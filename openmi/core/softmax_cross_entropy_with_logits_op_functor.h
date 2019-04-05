#ifndef OPENMI_CORE_OPS_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_OP_FUNCTOR_H_
#define OPENMI_CORE_OPS_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_OP_FUNCTOR_H_ 

#include "softmax_op_functor.h"
#include "base/logging.h"

namespace openmi {
namespace functor {

template <typename Device, typename T>
struct SoftmaxCrossEntropyWithLogitsFunctor {
  // logits: dim [batch_size, num_classes];
  // softmax_cross_entropy: dim [batch_size, num_classes]
  void operator()(const Device& d, typename TTypes<T>::Matrix labels, typename TTypes<T>::Matrix logits, typename TTypes<T>::Matrix loss);
}; // struct SoftmaxCrossEntropyWithLogits

/*!
 * implementing SoftmaxCrossEntropy using LogSoftmaxFunctor::operator
 */
template <typename Device, typename T>
struct SoftmaxCrossEntropyWithLogitsImpl {
  static void Compute(const Device& d, typename TTypes<T>::Matrix labels, typename TTypes<T>::Matrix logits, typename TTypes<T>::Matrix loss) {
    const int kBatchDim = 0;
    const int kClassDim = 1;
    // softmax predictor, not log_softmax
    typename TTypes<T>::Matrix softmax(labels.data(), 
                              labels.dimension(kBatchDim), 
                              labels.dimension(kClassDim));
    SoftmaxImpl<Device, T>::Compute(d, logits, softmax, false);

    LOG(DEBUG) << "softmax:\n" << softmax; 
    // softmax cross entropy loss 
    Eigen::DSizes<int, 1> along_class(kClassDim);
    loss.device(d) = (- labels * softmax.log()).sum(along_class);
    LOG(DEBUG) << "loss:\n" << loss; 
  }
}; // struct SoftmaxCrossEntropyWithLogitsImpl

} // namespace functor
} // namespace openmi
#endif // OPENMI_CORE_OPS_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_OP_FUNCTOR_H_
