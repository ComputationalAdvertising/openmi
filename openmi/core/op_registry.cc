#include "op_registry.h"
#include "base/logging.h"

namespace openmi {

OpRegistrationEntry* OpRegistrationData::FindEntry(const std::string device) {
  for (auto&& entry: entrys) {
    if (entry->device != device) {
      continue;
    }
    return entry;
  }
  return nullptr;
}

void OpRegistry::RegisterOp(std::string name, OpRegistrationEntry* entry) {
  //LOG(INFO) << "OpRegistry::RegisterOp name: " << name;
  auto iter = op_kernel_mapper_.find(name);
  if (iter != op_kernel_mapper_.end()) {
    CHECK(iter->second.FindEntry(entry->device) == nullptr) 
      << "<op_name, device> has already registry. op_name:" 
      << name << ", device:" << entry->device; 
    iter->second.entrys.insert(entry);
  } else {
    OpRegistrationData op_reg_data;
    op_reg_data.entrys.insert(entry);
    op_kernel_mapper_.insert({name, op_reg_data});
  }
}

Status OpRegistry::LookUp(const proto::NodeDef& node_def, OpKernel** op_kernel) {
  auto iter = op_kernel_mapper_.find(node_def.op());
  CHECK(iter != op_kernel_mapper_.end())
    << node_def.op() 
    << " not in op registry. please check whether 'op' field in node_def.proto is or not correct.";

  LOG(INFO) << "[TMP] op_name exists. " << node_def.op();
  
  OpRegistrationEntry* entry = iter->second.FindEntry(node_def.device());
  if (entry == nullptr) {
    entry = iter->second.FindEntry();
  }
  CHECK(entry != nullptr)
    << "device of op not exist. op_name:" 
    << node_def.op() << ", device:" << node_def.device();

  *op_kernel = entry->body();

  return Status::OK();
}

OpRegistryHelper::OpRegistryHelper(std::string op_name)
  : name_(op_name), entry_(new OpRegistrationEntry()) {}

OpRegistryHelper& OpRegistryHelper::SetBody(const std::function<OpKernel*()>& body) {
  entry_->body = body;
  return *this;
}

OpRegistryHelper& OpRegistryHelper::Device(std::string device) {
  entry_->device = device;
  return *this;
}

OPENMI_REGISTER_LINK_TAG(UnaryOp);

} // namespace openmi
