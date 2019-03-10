#include "gradient_op_registry.h"
#include "base/logging.h"

namespace openmi {

bool GradientOpRegistry::Register(std::string name, GradFunc grad_func) {
  auto it = gradient_op_mapper_.find(name);
  CHECK(it == gradient_op_mapper_.end()) 
    << "gradient op already exists. name:" << name;
  gradient_op_mapper_.insert({name, grad_func});
  LOG(INFO) << "register gradient op done. name: " << name;
  return true;
}

Status GradientOpRegistry::LookUp(std::string name, GradFunc** grad_op) {
  auto it = gradient_op_mapper_.find(name);
  CHECK(it != gradient_op_mapper_.end()) 
    << "gradient op not exists. name:" << name;
  *grad_op = &it->second;
  LOG(INFO) << "LookUp gradient op done. name: " << name;
  return Status::OK();
}

}
