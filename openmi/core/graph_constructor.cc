#include "graph_constructor.h"

#include <vector>
#include "base/logging.h"

namespace openmi {

class GraphConstructor {
public:
  static Status Construct(proto::GraphDef* gdef, Graph* g, 
                          std::vector<std::pair<Node*, int>>* return_tensors);

private:
  GraphConstructor(proto::GraphDef* gdef, Graph* g, 
                   std::vector<std::pair<Node*, int>>* return_tensors) 
    : gdef_(gdef), g_(g),
      return_tensors_(return_tensors) {}

  Status TryImport(); 

private:
  proto::GraphDef* gdef_;
  Graph* g_;
  // May be null. Not owned. 
  std::vector<std::pair<Node*, int>>* return_tensors_;
}; // class GraphConstructor

Status GraphConstructor::Construct(proto::GraphDef* gdef, 
                                   Graph* g, 
                                   std::vector<std::pair<Node*, int>>* return_tensors) {
  GraphConstructor gc(gdef, g, return_tensors);
  return gc.TryImport();
}

Status GraphConstructor::TryImport() {
  LOG(INFO) << "GraphConstructor::TryImport begin";
  g_->set_name(*(gdef_->mutable_name()));
  g_->set_version(gdef_->version());
  Status s;
  for (int i = 0; i < gdef_->node().size(); ++i) {
    proto::NodeDef* node_def = const_cast<proto::NodeDef*>(&gdef_->node(i));
    NodeInfo ninfo(*node_def, i, NC_OP, NS_FORWARD);
    g_->AddNode(ninfo, &s);
    LOG(INFO) << gdef_->node(i).name() << " node add done.";
  }

  // parse node dependencies 
  for (auto& node_def: gdef_->node()) {
    for (int i = 0; i < node_def.input().size(); ++i) {
      auto input = node_def.input(i);
      Node* n = g_->FindNode(input);
      CHECK(n != nullptr) << input << " not in graphdef.";
      g_->AddInput(node_def.name(), n);
    }
  }
  //LOG(INFO) << "GraphConstructor::TryImport done";
  return Status::OK();
}

extern Status ConvertGraphDefToGraph(GraphDef* gdef, Graph* g) {
  return GraphConstructor::Construct(gdef, g, nullptr);
}

}
