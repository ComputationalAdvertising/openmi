#ifndef OPENMI_CORE_OPS_ACCUMULATE_N_FUNCTOR_H_
#define OPENMI_CORE_OPS_ACCUMULATE_N_FUNCTOR_H_ 

#include <vector>
#include <Eigen/Core>
#include "tensor_types.h"
#include "base/logging.h"

using namespace openmi;

namespace openmi {
namespace functor {

template <typename Device, typename T>
struct AccumulateNImpl {
  static void Compute(const Device& d, std::vector<typename TTypes<T>::Flat > inputs, typename TTypes<T>::Flat out) {
    out.device(d) = inputs.at(0);
    for (auto i = 1; i < inputs.size(); ++i) {
      out.device(d) = out + inputs.at(i);
    }
  }
}; // struct AccumulateNFunctor

} // namespace functor
} // namespace openmi
#endif // OPENMI_CORE_OPS_ACCUMULATE_N_FUNCTOR_H_
