#ifndef OPENMI_CORE_OPS_SOFTMAX_OP_H_
#define OPENMI_CORE_OPS_SOFTMAX_OP_H_ 

#include "numeric_op.h"
#include "base/logging.h"

namespace openmi {

/*!
 * \brief softmax operator. 
 *   input: m * n 
 *   output: m * k
 *   m: batch size 
 *   n: vector represent
 *   k: number of class (default = 2)
 */
template <typename Device, typename T>
class SoftmaxOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) override {
    LOG(DEBUG) << "SoftmaxOp compute ...";
    // TODO
  }
}; // class SoftmaxOp

} // namespace openmi
#endif // OPENMI_CORE_OPS_SOFTMAX_OP_H_
