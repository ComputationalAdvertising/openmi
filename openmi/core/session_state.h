#ifndef OPENMI_CORE_FRAMEWORK_SESSION_STATE_H_
#define OPENMI_CORE_FRAMEWORK_SESSION_STATE_H_ 

#include <mutex>

#include "tensor.h"
#include "status.h"

namespace openmi {

class SessionState {
public:
  Status GetTensor(const std::string& handle, Tensor** tensor);

  Tensor& GetTensor(const std::string& handle);

  Status AddTensor(const std::string& handle, Tensor* tensor);

  Status DeleteTensor(const std::string& handle);

private:
  std::mutex mutex_;
  // The live tensors in the session. A map from tensor handle to tensor 
  std::unordered_map<std::string, Tensor*> tensor_mapper_;
}; // class SessionState

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_SESSION_STATE_H_
