#include "session_state.h"

namespace openmi {

Status SessionState::GetTensor(const std::string& handle, Tensor** tensor) {
  *tensor = &GetTensor(handle);
  return Status::OK();
}

Tensor& SessionState::GetTensor(const std::string& handle) {
  LOG(DEBUG) << "handler: " << handle;
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = tensor_mapper_.find(handle);
  CHECK(it != tensor_mapper_.end()) << " The Tensor with handle '" << handle 
    << "' is not in the session state.";
  return *it->second;
}

Status SessionState::AddTensor(const std::string& handle, Tensor* tensor) {
  std::unique_lock<std::mutex> lock(mutex_);
  CHECK(tensor_mapper_.insert({handle, tensor}).second) 
    << "Failed to add a tensor with handle '" << handle << "' to the session state.";
  return Status::OK();
}

Status SessionState::DeleteTensor(const std::string& handle) {
  std::unique_lock<std::mutex> lock(mutex_);
  CHECK(tensor_mapper_.erase(handle) != 0) 
    << "Failed to delete a tensor with handle '" << handle << "' in the session state.";
  return Status::OK();
}

}
