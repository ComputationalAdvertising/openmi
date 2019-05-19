#ifndef OPENMI_CORE_OPS_CROSS_ENTROPY_OP_FUNCTOR_H_
#define OPENMI_CORE_OPS_CROSS_ENTROPY_OP_FUNCTOR_H_ 

#include <Eigen/Core>

#include "softmax_op_functor.h"
#include "tensor_types.h"
#include "base/logging.h"
using namespace openmi;

namespace openmi {
namespace functor {

template <typename Device, typename T>
struct SoftmaxCrossEntropyWithLogitsFunctor {
  // logits: dim [batch_size, num_classes];
  // softmax_cross_entropy: dim [batch_size, num_classes]
  void operator()(const Device& d, typename TTypes<T>::Matrix labels, typename TTypes<T>::Matrix logits, typename TTypes<T>::Matrix loss);
}; // struct SoftmaxCrossEntropyWithLogits

template <typename Device, typename T>
struct SoftmaxCrossEntropyFunctor {
  void operator()(const Device& d, typename TTypes<T>::Matrix labels, typename TTypes<T>::Matrix softmax, typename TTypes<T>::Matrix loss);
}; // struct SoftmaxCrossEntropyFunctor

/*!
 * implementing SoftmaxCrossEntropy
 */
template <typename Device, typename T>
struct SoftmaxCrossEntropyImpl {
  static void Compute(const Device& d, 
                      typename TTypes<T>::Matrix labels, 
                      typename TTypes<T>::Matrix softmax,
                      typename TTypes<T>::Matrix loss) {
    const int kBatchDim = 0;
    const int kClassDim = 1;
    Eigen::DSizes<int, 1> along_class(kClassDim);
    loss.device(d) = (- labels * softmax.log()).sum(along_class);
  }
}; // struct SoftmaxCrossEntropyImpl

/*!
 * implementing SoftmaxCrossEntropyWithLogits using SoftmaxFunctor::operator
 */
template <typename Device, typename T>
struct SoftmaxCrossEntropyWithLogitsImpl {
  static void Compute(const Device& d, 
                      typename TTypes<T>::Matrix labels, 
                      typename TTypes<T>::Matrix logits,
                      typename TTypes<T>::Matrix loss) {
    const int kBatchDim = 0;
    const int kClassDim = 1;
    // softmax predictor, not log_softmax
    typename TTypes<T>::Matrix softmax(logits.data(), 
                                       logits.dimension(kBatchDim), 
                                       logits.dimension(kClassDim));
    SoftmaxImpl<Device, T>::Compute(d, logits, softmax, false);
    // softmax cross entropy loss  
    SoftmaxCrossEntropyImpl<Device, T>::Compute(d, labels, softmax, loss);
    LOG(DEBUG) << "loss:\n" << loss; 
  }
}; // struct SoftmaxCrossEntropyWithLogitsImpl 

template <typename Device, typename T>
struct SigmoidCrossEntropyImpl {
  static void Compute(const Device& d, 
                      typename TTypes<T>::Matrix labels, 
                      typename TTypes<T>::Matrix sigmoid, 
                      typename TTypes<T>::Matrix loss) {
    loss.device(d) = - labels * sigmoid.log() - (1 - labels) * (1 - sigmoid).log();
  }
}; // struct SigmoidCrossEntropyImpl

} // namespace functor
} // namespace openmi
#endif // OPENMI_CORE_OPS_CROSS_ENTROPY_OP_FUNCTOR_H_
