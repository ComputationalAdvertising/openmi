#ifndef OPENMI_CORE_FRAMEWORK_ATTR_VALUE_H_
#define OPENMI_CORE_FRAMEWORK_ATTR_VALUE_H_ 

#include <string>
#include "tensor.h"
#include "tensor_shape.h"
#include "openmi/idl/proto/node_def.pb.h"

namespace openmi {

struct AttrValue {
  enum AttrType {
    kNone,
    kString,
    kInt,
    kFloat,
    kBool,
    kType,
    kShape,
    kTensor,
  };

  AttrType attr_type;
  std::string s;
  int64_t i;
  float f;
  bool b;
  proto::DataType type;
  TensorShape shape; 
  Tensor tensor;

  void FromProto(const proto::AttrValue& value);
  proto::AttrValue ToProto();
}; // class AttrValue

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_ATTR_VALUE_H_ 
