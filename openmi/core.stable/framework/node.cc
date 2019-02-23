#include "core/framework/node.h"
#include "openmi/pb/tensor_shape.pb.h"

namespace openmi {

Node::Node(pb::Node& node_def) {
  Op* op = openmi::Register<OpFactory>::Find(node_def.op())->func();
  if (op == nullptr) {
    LOG(ERROR) << "op is null. op_name:" << node_def.op();
    return;
  }
  op_ = op;
  name_ = node_def.name();
  id_ = node_def.id();
  type_ = NT_UNINITIALIZED;
  compute_type_ = NCT_NOTHING;
  value_ = 0;
  pb::TensorShapeProto shape = node_def.attr().shapes();
  LOG(INFO) << "shapes.size()=" << shape.shape().size() << ", name: " << name_;
  tensor_ = new openmi::Tensor<float>(shape);
}

Node::Node(std::string name, int id, const TensorShape& shape) 
  : name_(name), id_(id), type_(NT_UNINITIALIZED), compute_type_(NCT_NOTHING) {
    value_ = 0;
    tensor_ = new openmi::Tensor<float>(shape);
}

}
