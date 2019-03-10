#include "session_state.h"
#include "graph_constructor.h"
#include "base/protobuf_op.h"
#include "base/logging.h"
#include "graph.h"

int main(int argc, char** argv) {
  const char* file = "conf/graph_demo.conf";
  proto::GraphDef gdef;
  if (ProtobufOp::LoadObjectFromPbFile<proto::GraphDef>(file, &gdef) != 0) {
    LOG(ERROR) << "load graph def proto file failed.";
    return -1;
  }

  Graph* g = new Graph();
  Status s = ConvertGraphDefToGraph(&gdef, g);

  SessionState session_state;
  for (auto& kv: g->node_mapper()) {
    LOG(INFO) << "node name: " << kv.first;
    DataType type = DT_FLOAT;
    auto it = kv.second->attrs().find("type");
    if (it != kv.second->attrs().end()) {
      CHECK(it->second.attr_type == ::openmi::AttrValue::kType);
      type = it->second.type;
    }

    it = kv.second->attrs().find("shape");
    if (it != kv.second->attrs().end()) {
      CHECK(it->second.attr_type == ::openmi::AttrValue::kShape);
      Tensor tensor(type, it->second.shape);
      session_state.AddTensor(kv.second->def().name(), &tensor);
    }

  }
  LOG(INFO) << "session_state AddTensor done";

  Tensor* t = nullptr;
  session_state.GetTensor("w", &t);
  LOG(INFO) << "session_state AddTensor done";
  LOG(INFO) << "shape(w): " << t->shape().DebugString();
  LOG(INFO) << "session_state AddTensor done";
  return 0;
}
