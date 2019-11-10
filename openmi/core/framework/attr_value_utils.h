#ifndef OPENMI_CORE_FRAMEWORK_ATTR_VALUE_UTILS_H_
#define OPENMI_CORE_FRAMEWORK_ATTR_VALUE_UTILS_H_ 

#include "openmi/idl/proto/node_def.pb.h"
#include "openmi/idl/proto/types.pb.h"
#include "attr_value.h"

using namespace openmi::proto;

namespace openmi {

inline proto::AttrValue* attr_s(std::string s) {
  proto::AttrValue* attr = new proto::AttrValue();
  attr->set_s(s);
  return attr;
}

inline proto::AttrValue* attr_b(bool b) {
  proto::AttrValue* attr = new proto::AttrValue();
  attr->set_b(b);
  return attr;
}

inline proto::AttrValue* attr_i(int i) {
  proto::AttrValue* attr = new proto::AttrValue();
  attr->set_i(i);
  return attr;
}

inline proto::AttrValue* attr_f(float f) {
  proto::AttrValue* attr = new proto::AttrValue();
  attr->set_f(f);
  return attr;
}

inline proto::AttrValue* attr_type(DataType type) {
  proto::AttrValue* attr = new proto::AttrValue();
  attr->set_type(type);
  return attr;
}

inline proto::AttrValue* attr_shape(std::vector<int> dims) {
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

inline void GetAttr(std::unordered_map<std::string, AttrValue>& attr, const std::string& key, bool* v, bool allow_not_found = true) {
  auto it = attr.find(key);
  if (!allow_not_found) {
    CHECK(it != attr.end()) << "attr '" << key << "' not exists. please check graph def.";
  }
  if (it != attr.end()) {
    *v = it->second.b;
  }
}

inline void GetAttr(std::unordered_map<std::string, AttrValue>& attr, const std::string& key, int* v, bool allow_not_found = true) {
  auto it = attr.find(key);
  if (!allow_not_found) {
    CHECK(it != attr.end()) << "attr '" << key << "' not exists. please check graph def.";
  }
  if (it != attr.end()) {
    *v = it->second.i;
  }
}

inline void GetAttr(std::unordered_map<std::string, AttrValue>& attr, const std::string& key, float* v, bool allow_not_found = true) {
  auto it = attr.find(key);
  if (!allow_not_found) {
    CHECK(it != attr.end()) << "attr '" << key << "' not exists. please check graph def.";
  }
  if (it != attr.end()) {
    *v = it->second.f;
  }
}


inline void GetAttr(std::unordered_map<std::string, AttrValue>& attr, const std::string& key, std::string* v, bool allow_not_found = true) {
  auto it = attr.find(key);
  if (!allow_not_found) {
    CHECK(it != attr.end()) << "attr '" << key << "' not exists. please check graph def.";
  }
  if (it != attr.end()) {
    *v = it->second.s;
  }
}

inline void GetAttr(std::unordered_map<std::string, AttrValue>& attr, const std::string& key, proto::DataType* v, bool allow_not_found = true) {
  auto it = attr.find(key);
  if (!allow_not_found) {
    CHECK(it != attr.end()) << "attr '" << key << "' not exists. please check graph def.";
  }
  if (it != attr.end()) {
    *v = it->second.type;
  }
}

inline void GetAttr(std::unordered_map<std::string, AttrValue>& attr, const std::string& key, TensorShape* v, bool allow_not_found = true) {
  auto it = attr.find(key);
  if (!allow_not_found) {
    CHECK(it != attr.end()) << "attr '" << key << "' not exists. please check graph def.";
  }
  if (it != attr.end()) {
    *v = it->second.shape;
  }
}

inline void GetAttr(std::unordered_map<std::string, AttrValue>& attr, const std::string& key, Tensor* v, bool allow_not_found = true) {
  auto it = attr.find(key);
  if (!allow_not_found) {
    CHECK(it != attr.end()) << "attr '" << key << "' not exists. please check graph def.";
  }
  if (it != attr.end()) {
    *v = it->second.tensor;
  }
}

inline void GetAttr(std::unordered_map<std::string, AttrValue>& attr, const std::string& key, proto::FeatureType* v, bool allow_not_found = true) {
  auto it = attr.find(key);
  if (!allow_not_found) {
    CHECK(it != attr.end()) << "attr '" << key << "' not exists. please check graph def.";
  }
  if (it != attr.end()) {
    *v = it->second.feature_type;
  }
}

inline void GetAttr(std::unordered_map<std::string, AttrValue>& attr, const std::string& key, proto::SourceNodeType* v, bool allow_not_found = true) {
  auto it = attr.find(key);
  if (!allow_not_found) {
    CHECK(it != attr.end()) << "attr '" << key << "' not exists. please check graph def.";
  }
  if (it != attr.end()) {
    *v = it->second.source_node_type;
  }
}

inline void GetAttr(std::unordered_map<std::string, AttrValue>& attr, const std::string& key, proto::OptimizerConfig* v, bool allow_not_found = true) {
  auto it = attr.find(key);
  if (!allow_not_found) {
    CHECK(it != attr.end()) << "attr '" << key << "' not exists. please check graph def.";
  }
  if (it != attr.end()) {
    *v = it->second.optimizer;
  }
}

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_ATTR_VALUE_UTILS_H_
