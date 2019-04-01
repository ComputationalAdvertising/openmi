#ifndef OPENMI_CORE_FRAMEWORK_ATTR_VALUE_UTILS_H_
#define OPENMI_CORE_FRAMEWORK_ATTR_VALUE_UTILS_H_ 

#include "openmi/idl/proto/node_def.pb.h"
#include "openmi/idl/proto/types.pb.h"

using namespace openmi::proto;

namespace openmi {

proto::AttrValue* attr_s(std::string s) {
  proto::AttrValue* attr = new proto::AttrValue();
  attr->set_s(s);
  return attr;
}

proto::AttrValue* attr_b(bool b) {
  proto::AttrValue* attr = new proto::AttrValue();
  attr->set_b(b);
  return attr;
}

proto::AttrValue* attr_i(int i) {
  proto::AttrValue* attr = new proto::AttrValue();
  attr->set_i(i);
  return attr;
}

proto::AttrValue* attr_f(float f) {
  proto::AttrValue* attr = new proto::AttrValue();
  attr->set_f(f);
  return attr;
}

proto::AttrValue* attr_type(DataType type) {
  proto::AttrValue* attr = new proto::AttrValue();
  attr->set_type(type);
  return attr;
}

proto::AttrValue* attr_shape(std::vector<int> dims) {
  proto::AttrValue* attr = new proto::AttrValue();
  auto shape = attr->mutable_shape();
  for (int i = 0; i < dims.size(); ++i) {
    int dim_size = dims[i];
    auto dim = shape->add_dim();
    dim->set_size(dim_size);
    dim->set_name("dim_" + std::to_string(i+1));
  }
  return attr;
}



} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_ATTR_VALUE_UTILS_H_
