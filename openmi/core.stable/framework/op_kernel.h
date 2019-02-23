#ifndef OPENMI_CORE_FRAMEWORK_OP_KERNEL_H_
#define OPENMI_CORE_FRAMEWORK_OP_KERNEL_H_ 

#include "core/framework/device.h"
#include "core/lib/status.h"

namespace openmi {

class OpKernelConstruction;
class OpkernelContext;

class OpKernel {
public:
  explicit OpKernel(OpKernelConstruction* ctx);
  virtual ~OpKernel();

  // All OpKernel Compute() methods must be thread-safe as they 
  // may be called concurrently. 
  // "context " is guaranteed to be alive until Compute() returns. 
  virtual void Compute(OpKernelContext* ctx) = 0;

private:
  //std::unique_ptr<NodeDef> node_def_;
}; // class OpKernel
   
// TODO OpKernelAsync 

class OpKernelContext {
public:
  struct Params {
    int64 step_id = 0;
    Device* device = nullptr;

    OpKernel* op_kernel = nullptr;
    // The inputs for this op 
    std::vector<std::string> input_name;      // node_def.name 
    
    // The outputs for this op 
    std::vector<std::string> output_name;

    // The session state for this op. 
    SessionState* session_state = nullptr;
    TensorStore* tensor_store = nullptr;
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
