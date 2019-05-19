#ifndef OPENMI_CORE_OPS_VARIABLE_OP_H_
#define OPENMI_CORE_OPS_VARIABLE_OP_H_ 

#include "op_kernel.h"

namespace openmi {

class VariableOp : public OpKernel {
public:
  void Initialize(OpKernelConstruction* context) override {
    LOG(INFO) << "VariableOp init";
  }

  void Compute(OpKernelContext* context) override {
    LOG(INFO) << "VariableOp compute";
  }
}; // class VariableOp 

}
#endif // OPENMI_CORE_OPS_VARIABLE_OP_H_
