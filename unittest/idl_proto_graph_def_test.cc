#include "openmi/idl/proto/graph.pb.h"
#include "openmi/idl/proto/node_def.pb.h"
#include "openmi/idl/proto/types.pb.h"
#include "base/protobuf_op.h"

using namespace openmi;
using namespace openmi::proto;

proto::AttrValue* attr_s(std::string s) {
  proto::AttrValue* attr = new proto::AttrValue();
  attr->set_s(s);
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

int main(int argc, char** argv) {
  const char* file = "conf/graph_demo.conf";
  proto::GraphDef gdef;
  if (ProtobufOp::LoadObjectFromPbFile<proto::GraphDef>(file, &gdef) != 0) {
    LOG(ERROR) << "load graph def proto file failed.";
    return -1;
  }
  
  LOG(INFO) << "graph_demo.proto:\n" << gdef.DebugString();

  auto attr = const_cast<proto::NodeDef&>(gdef.node(0)).mutable_attr();
  attr->insert({"test", *attr_s("test.value")});
  attr->insert({"type", *attr_type(DT_FLOAT)});
  std::vector<int> dims;
  dims.push_back(100);
  dims.push_back(10000);
  dims.push_back(10000000);
  attr->insert({"shape1", *attr_shape(dims)});
  
  LOG(INFO) << "updated graph_demo.proto:\n" << gdef.DebugString();
  return 0;
}
