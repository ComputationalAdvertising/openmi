#ifndef OPENMI_CORE_FRAMEWORK_OP_DEFINE_BUILDER_H_
#define OPENMI_CORE_FRAMEWORK_OP_DEFINE_BUILDER_H_ 

#include <string>
#include <vector>
#include "core/lib/status.h"
#include "openmi/pb/op_def.pb.h"

namespace openmi {

struct OpRegistrationData {
  OpRegistrationData() {}
  OpRegistrationData(const OpDef& def): op_def(def) {}

  OpDef op_def;
}; // struct OpRegistrationData

// Builder class passed to the REGISTER_OP() macro 
class OpDefBuilder {
public:
  explicit OpDefBuilder(std::string op_name);

  // Adds an attr to this OpDefBuilder and return *this. 
  OpDefBuilder& Attr(std::string spec);

  // Adds an input or output to this OpDefBuilder and return *this.
  OpDefBuilder& Input(std::string spec);
  OpDefBuilder& Output(std::string spec); 

  // Turns on the indicated boolean flag in this OpDefBuilder and return *this. 
  OpDefBuilder& SetIsCommutative();
  OpDefBuilder& SetIsAggregate();
  OpDefBuilder& SetIsStateful();
  OpDefBuilder& SetAllowsUninitializedInput(); 

  // Sets op_reg_data->op_def to the requested OpDef 
  Status Finalize(OpRegistrationData* op_reg_data) const;

private:
  OpDef* op_def() { return &op_reg_data_.op_def; }

  OpRegistrationData op_reg_data_;
  std::vector<std::string> attrs_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  std::string doc_;
  std::vector<std::string> errors_;
}; // class OpDefBuilder

} // namespace openmi
#endif  // OPENMI_CORE_FRAMEWORK_OP_DEFINE_BUILDER_H_
