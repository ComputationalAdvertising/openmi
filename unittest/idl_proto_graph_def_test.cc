#include "openmi/idl/proto/graph.pb.h"
#include "openmi/idl/proto/node_def.pb.h"
#include "openmi/idl/proto/types.pb.h"
#include "base/protobuf_op.h"
#include "attr_value_utils.h"

using namespace openmi;
using namespace openmi::proto;

int main(int argc, char** argv) {
  const char* file = "unittest/conf/graph_demo.conf";
  proto::GraphDef gdef;
  if (ProtobufOp::LoadObjectFromPbFile<proto::GraphDef>(file, &gdef) != 0) {
    LOG(ERROR) << "load graph def proto file failed.";
    return -1;
  }
  
  LOG(INFO) << "graph_demo.proto gdef.node[0]:\n" << gdef.node(0).attr().size();

  auto attr = const_cast<proto::NodeDef&>(gdef.node(0)).mutable_attr();
  attr->insert({"test", *attr_s("test.value")});
  attr->insert({"type", *attr_type(DT_FLOAT)});
  std::vector<int> dims;
  dims.push_back(100);
  dims.push_back(10000);
  dims.push_back(10000000);
  attr->insert({"shape1", *attr_shape(dims)});
  
  std::string debug_string = gdef.DebugString();
  LOG(INFO) << "updated graph_demo.proto:\n" << debug_string;
  LOG(INFO) << "debug_string.size:" << debug_string.size();

  std::string serialized_string;
  //gdef.SerializeToString(&serialized_string);
  ProtobufOp::SerializeToString<proto::GraphDef>(&gdef, serialized_string);
  LOG(INFO) << "serialized_string.size: " << serialized_string.size();
  LOG(INFO) << "gdef.length: " << gdef.ByteSizeLong();

  const char* tmp_file = "text.tmp";
  ProtobufOp::SerializeToOstream<proto::GraphDef>(tmp_file, &gdef);
  ProtobufOp::ParseFromIstream<proto::GraphDef>(tmp_file, &gdef);
  LOG(INFO) << gdef.ByteSizeLong();
  return 0;
}
