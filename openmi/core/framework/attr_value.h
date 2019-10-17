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
    kFeaturetype,
    kSourceNodeType,
    kOptimizer
  };

  AttrType attr_type;
  std::string s;
  int64_t i;
  float f;
  bool b;
  proto::DataType type;
  TensorShape shape; 
  Tensor tensor;
  proto::FeatureType feature_type;
  proto::SourceNodeType source_node_type;
  proto::Optimizer optimizer;

  void FromProto(const proto::AttrValue& value);
  proto::AttrValue ToProto();
}; // class AttrValue

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_ATTR_VALUE_H_ 
