#ifndef OPENMI_CORE_OPS_SIGMOID_OP_H_
#define OPENMI_CORE_OPS_SIGMOID_OP_H_ 

#include <Eigen/Core>
#include "tensor_types.h"
using namespace openmi;
#include "base/logging.h"

namespace openmi {
namespace functor {

template <typename Device, typename T>
struct SigmoidImpl {
  static void Compute(const Device& d, typename TTypes<T>::Matrix logits, typename TTypes<T>::Matrix sigmoid) {
    sigmoid.device(d) = (static_cast<T>(1) + (-logits).exp()).inverse();
  }
}; // struct SigmoidImpl

} // namespace functor
} // namespace openmi
#endif // OPENMI_CORE_OPS_SIGMOID_OP_H_
