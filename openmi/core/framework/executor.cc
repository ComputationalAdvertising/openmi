#include "core/framework/executor.h"

using namespace openmi::graph;

namespace openmi {

Executor::Executor(std::vector<Node*> nodes_to_be_computed): 
  nodes_to_be_computed_(nodes_to_be_computed) {
  CHECK(TopoOrderList(nodes_to_be_computed_, topo_order_list_) == 0) 
    << "get topo order list failed.";
}

Executor::~Executor() {
}

int Executor::Run() {
  for (size_t i = 0; i < topo_order_list_.size(); ++i) {
    Node* node = topo_order_list_[i];
    if (node->GetOp()->Name() == "PlaceholderOp") {
      continue;
    }
    std::vector<Node*> inputs = node->Inputs();
    node->GetOp()->Compute(node, inputs);
  }
  return 0;
}

} // namespace openmi
