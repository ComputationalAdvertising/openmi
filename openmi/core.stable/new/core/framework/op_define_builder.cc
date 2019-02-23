#include "core/framework/op_define_builder.h"

namespace openmi {

void FinalizeAttr(std::string spec, OpDef* op_def, std::vector<std::string>* errors) {
  OpDef::AttrDef* attr = op_def->add_attr();
  attr->set_name(spec);

  
}

OpDefBuilder::OpDefBuilder(std::string op_name) {
  op_def()->set_name(op_name);
}

OpDefBuilder& OpDefBuilder::Attr(std::string spec) {
  attrs_.emplace_back(spec);
  return *this;
}

OpDefBuilder& OpDefBuilder::Input(std::string spec) {
  inputs_.emplace_back(spec);
  return *this;
}

OpDefBuilder& OpDefBuilder::Output(std::string spec) {
  outputs_.emplace_back(spec);
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsCommutative() {
  op_def()->set_is_commutative(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsAggregate() {
  op_def()->set_is_aggregate(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsStateful() {
  op_def()->set_is_stateful(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::SetAllowsUninitializedInput() {
  op_def()->set_allows_uninitialized_input(true);
  return *this;
}

Status OpDefBuilder::Finalize(OpRegistrationData* op_reg_data) const {
  std::vector<std::string> errors = errors_;
  *op_reg_data = op_reg_data_;

  OpDef* op_def = &op_reg_data->op_def;
  for (std::string attr: attrs_) {
    FinalizeAttr(attr, op_def, &errors);
  }
  for (std::string input: inputs_) {
    FinalizeInputOrOutput(input, false, op_def, &errors);
  }
  for (std::string output: outputs_) {
    FinalizeInputOrOutput(output, true, op_def, &errors);
  }
  // FinalizeDoc 

  if (errors.empty()) return Status::OK();
  return Status(INVALID_ARGUMENT, "invalid argument");
}


}
