#ifndef OPENMI_CORE_FRAMEWORK_OP_KERNEL_H_
#define OPENMI_CORE_FRAMEWORK_OP_KERNEL_H_ 

#include "core/framework/device.h"
#include "core/lib/status.h"

namespace openmi {

class OpKernel {
  // TODO compute
}; // class OpKernel
   
// TODO OpKernelAsync 

class OpKernelContext {
public:
  struct Params {
    Device* device = nullptr;
  };

  explicit OpKernelContext(Params* params);

  ~OpKernelContext();

  Status Allocate(TensorShape& shape, DataType type);

  Device* GetDevice() const { 
    return params_->device; 
  }

private:
  Params* params_;
}; // class OpKernelContext


} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_OP_KERNEL_H_
