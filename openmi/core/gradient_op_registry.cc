#include "gradient_op_registry.h"
#include "base/logging.h"
#include "base/register.h"

namespace openmi {

void GradientOpRegistry::RegisterOp(std::string name, OpRegistrationEntry* entry) {
  LOG(INFO) << "GradientOpRegistry::RegisterOp name: " << name;
  auto iter = gradient_op_mapper_.find(name);
  if (iter != gradient_op_mapper_.end()) {
    CHECK(iter->second.FindEntry(entry->device, entry->allow_type) == nullptr) 
      << "<op_name, device> has already registry. op_name:" 
      << name << ", device:" << entry->device; 
    iter->second.entrys.insert(entry);
  } else {
    OpRegistrationData op_reg_data;
    op_reg_data.entrys.insert(entry);
    gradient_op_mapper_.insert({name, op_reg_data});
  }
}

Status GradientOpRegistry::LookUp(Node& node, OpKernel** grad_op) {
  auto iter = gradient_op_mapper_.find(node.def().op());
  CHECK(iter != gradient_op_mapper_.end())
    << node.def().op() 
    << " not in op registry. please check whether 'op' field in node_def.proto is or not correct.";

  
  DataType type = DT_FLOAT;
  auto it = node.attrs().find("type");
  if (it != node.attrs().end() 
      && it->second.attr_type == ::openmi::AttrValue::kType) {
    type = it->second.type;
  }
  OpRegistrationEntry* entry = iter->second.FindEntry(node.def().device(), type);
  CHECK(entry != nullptr)
    << "device of op not exist. op_name:" 
    << node.def().op() << ", device:" << node.def().device();

  *grad_op = static_cast<OpKernel*>(entry->body());

  LOG(INFO) << "LookUp gradient op successfully. op_name:" << node.def().op();

  return Status::OK();
}

OPENMI_REGISTER_LINK_TAG(matmul_grad_op);

}
