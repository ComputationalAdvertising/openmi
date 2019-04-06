#ifndef OPENMI_CORE_FRAMEWORK_OP_KERNEL_H_
#define OPENMI_CORE_FRAMEWORK_OP_KERNEL_H_ 

#include <string>
#include <vector>
#include "attr_value.h"
#include "attr_value_utils.h"
#include "device.h"
#include "status.h"
#include "session_state.h"
#include "tensor_utils.h"
#include "openmi/idl/proto/node_def.pb.h"
#include "openmi/idl/proto/types.pb.h"
#include "base/logging.h"

using namespace openmi;
using namespace openmi::proto;

namespace openmi {

class OpKernelConstruction;
class OpKernelContext;

class OpKernel {
public:
  OpKernel();
  virtual ~OpKernel();

  virtual void Initialize(OpKernelConstruction* context);

  // All OpKernel Compute() methods must be thread-safe as they 
  // may be called concurrently. 
  // "context " is guaranteed to be alive until Compute() returns. 
  virtual void Compute(OpKernelContext* ctx) = 0;
}; // class OpKernel
  
// TODO AsyncOpKernel

class OpKernelConstruction {
public:
  OpKernelConstruction(
    const std::string& name,
    std::unordered_map<std::string, AttrValue>& attr)
  : name_(name), attr_(attr) {}

  std::unordered_map<std::string, AttrValue>& attrs() { return attr_; }

  template <typename T>
  void GetAttr(const std::string& key, T* value, AttrValue::AttrType attr_type) {
    openmi::GetAttr<T>(attr_, key, value, attr_type);
  }

  std::string name() { return name_; }

private:
  std::string name_; // node.name 
  std::unordered_map<std::string, AttrValue> attr_;
}; // class OpKernelConstruction

class OpKernelContext {
public:
  struct Params {
    //uint64_t step_id = 0;
    Device* device = nullptr;
    SessionState* session_state = nullptr;
    OpKernel* op_kernel = nullptr;
    proto::NodeDef* node_def = nullptr;
    // The inputs for this op 
    std::vector<std::string> input_name;      // node_def.name 
    // The outputs for this op 
    std::vector<std::string> output_name;
    std::string related_node_name;
  };

  explicit OpKernelContext(Params* params);

  ~OpKernelContext();

  OpKernel& op_kernel() { return *params_->op_kernel; }

  std::vector<std::string>& inputs() { return params_->input_name; }

  std::vector<std::string>& outputs() { return params_->output_name; }

  SessionState* session_state() { return params_->session_state; } 

  proto::NodeDef& node_def() { return *(params_->node_def); }

  std::string name() { return node_def().name(); }

  std::string related_node_name() { return params_->related_node_name; }

  Tensor& input(int index) {
    auto handle = params_->input_name.at(index);
    auto* t = params_->session_state->GetTensor(handle);
    CHECK(t != nullptr) << "handle '" << handle << "' not in session state.";
    return *t;
  }
  
  Tensor& output() {
    auto handle = params_->node_def->name();
    auto* t = params_->session_state->GetTensor(handle);
    CHECK(t != nullptr) << "handle '" << handle << "' not in session state.";
    return *t;
  }

  Tensor& output(int index) {
    auto handle = params_->output_name.at(index);
    auto* t = params_->session_state->GetTensor(handle);
    CHECK(t != nullptr) << " handle '" << handle << "' not in session_state.";
    return *t;
  }

  Tensor* GetTensor(const std::string& name) {
    return params_->session_state->GetTensor(name);
  }

  template <typename EigenDeviceType>
  const EigenDeviceType& eigen_device() const;

private:
  Params* params_;
}; // class OpKernelContext

typedef Eigen::ThreadPoolDevice CpuDevice;

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_OP_KERNEL_H_
