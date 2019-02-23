#ifndef OPENMI_CORE_FRAMEWORK_OP_KERNEL_H_
#define OPENMI_CORE_FRAMEWORK_OP_KERNEL_H_ 

#include <string>
#include <vector>
#include "status.h"

namespace openmi {

class OpKernelConstruction;
class OpKernelContext;

enum DataType {
  DT_UNVALID,
  DT_FLOAT,
  DT_INT
}; 

class OpKernel {
public:
  OpKernel();
  virtual ~OpKernel();

  virtual void Initialize(OpKernelConstruction* context);

  // All OpKernel Compute() methods must be thread-safe as they 
  // may be called concurrently. 
  // "context " is guaranteed to be alive until Compute() returns. 
  virtual void Compute(OpKernelContext* ctx) = 0;

protected:
  //std::unique_ptr<NodeDef> node_def_;
  OpKernelConstruction* context_;
  bool initialized_;
}; // class OpKernel
   
// TODO OpKernelAsync  

class OpKernelConstruction {
public:
  OpKernelConstruction(std::string name) : name_(name) {}

  std::string name_;
  std::vector<DataType> input_types_;
  std::vector<DataType> output_types_;
}; // class OpKernelConstruction

class OpKernelContext {
public:
  struct Params {
    uint64_t step_id = 0;

    OpKernel* op_kernel = nullptr;
    // The inputs for this op 
    std::vector<std::string> input_name;      // node_def.name 
    
    // The outputs for this op 
    std::vector<std::string> output_name;

  };

  explicit OpKernelContext(Params* params);

  ~OpKernelContext();

private:
  Params* params_;
}; // class OpKernelContext

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_OP_KERNEL_H_
