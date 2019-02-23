#include "graph_constructor.h"
#include "base/protobuf_op.h"
#include "base/logging.h"

int main(int argc, char** argv) {
  const char* file = "conf/graph_demo.conf";
  proto::GraphDef gdef;
  if (ProtobufOp::LoadObjectFromPbFile<proto::GraphDef>(file, &gdef) != 0) {
    LOG(ERROR) << "load graph def proto file failed.";
    return -1;
  }

  LOG(INFO) << "gdef:\n" << gdef.DebugString();

  Graph* g = new Graph();
  LOG(INFO) << g->version();
  Status s = ConvertGraphDefToGraph(&gdef, g);

  CHECK(gdef.node().size() == g->node_mapper().size()) << " number of node not match";
  return 0;
}
