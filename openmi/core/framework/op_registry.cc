#include "op_registry.h"
#include "base/logging.h"

namespace openmi {

OpRegistrationEntry* OpRegistrationData::FindEntry(const std::string device, DataType type) {
  for (auto&& entry: entrys) {
    if (entry->device != device) {
      continue;
    }
    if (entry->allow_type != type) {
      continue;
    }
    return entry;
  }
  return nullptr;
}

void OpRegistry::RegisterOp(std::string name, OpRegistrationEntry* entry) {
  std::string register_info = "op_name:" + name 
    + ", device:" + entry->device 
    + ", type:" + DataType_Name(entry->allow_type); 
  LOG(INFO) << "OpRegistry::RegisterOp name: " << register_info;
  auto iter = op_kernel_mapper_.find(name);
  if (iter != op_kernel_mapper_.end()) {
    CHECK(iter->second.FindEntry(entry->device, entry->allow_type) == nullptr) 
      << "<op_name, device> has already registry. " << register_info;
    iter->second.entrys.insert(entry);
  } else {
    OpRegistrationData op_reg_data;
    op_reg_data.entrys.insert(entry);
    op_kernel_mapper_.insert({name, op_reg_data});
  }
}

Status OpRegistry::LookUp(Node& node, OpKernel** op_kernel) {
  auto iter = op_kernel_mapper_.find(node.def().op());
  CHECK(iter != op_kernel_mapper_.end())
    << "op["<< node.def().op() << "] not in op registry. "
    << "please check whether 'op' field in node_def.proto is or not correct.";

  
  DataType type = DT_FLOAT;
  auto it = node.attrs().find("type");
  if (it != node.attrs().end() 
      && it->second.attr_type == ::openmi::AttrValue::kType) {
    type = it->second.type;
  }
  OpRegistrationEntry* entry = iter->second.FindEntry(node.def().device(), type);
  /*
  if (entry == nullptr) {
    entry = iter->second.FindEntry();
  }
  */
  CHECK(entry != nullptr)
    << "op info not exist. node_name[" << node.def().name() 
    << "], op_name[" << node.def().op() 
    << "], device[" << node.def().device() 
    << "], type[" << DataType_Name(type) << "]";

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

OpRegistryHelper& OpRegistryHelper::TypeConstraint(DataType type) {
  entry_->allow_type = type;
  return *this;
}

OPENMI_REGISTER_LINK_TAG(accumulate_n_op);
OPENMI_REGISTER_LINK_TAG(binary_add_op);
OPENMI_REGISTER_LINK_TAG(binary_sub_op);
OPENMI_REGISTER_LINK_TAG(binary_mul_op);
OPENMI_REGISTER_LINK_TAG(binary_div_op);
OPENMI_REGISTER_LINK_TAG(binary_sigmoid_grad_op);
OPENMI_REGISTER_LINK_TAG(cwise_ops_binary);
OPENMI_REGISTER_LINK_TAG(cwise_ops_unary);
OPENMI_REGISTER_LINK_TAG(concat_op);
OPENMI_REGISTER_LINK_TAG(matmul_op);
OPENMI_REGISTER_LINK_TAG(matmul_grad_op);
OPENMI_REGISTER_LINK_TAG(nothing_op);
OPENMI_REGISTER_LINK_TAG(oneslike_op);
OPENMI_REGISTER_LINK_TAG(reduce_sum_op);
OPENMI_REGISTER_LINK_TAG(register_ops);
OPENMI_REGISTER_LINK_TAG(segment_op);
OPENMI_REGISTER_LINK_TAG(sigmoid_op);
OPENMI_REGISTER_LINK_TAG(sigmoid_cross_entropy_with_logits);
OPENMI_REGISTER_LINK_TAG(slice_op);
OPENMI_REGISTER_LINK_TAG(softmax_cross_entropy_with_logits);
OPENMI_REGISTER_LINK_TAG(softmax_op);

} // namespace openmi
