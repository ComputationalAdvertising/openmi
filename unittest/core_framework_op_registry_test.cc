#include "gradient_registry.h"
#include "base/logging.h"

int main(int argc, char** argv) {
  proto::NodeDef node_def;
  node_def.set_op("Sigmoid");
  node_def.set_device("CPU");
  NodeInfo ninfo(node_def, -1, NC_UNINITIALIZED, NS_FORWARD);
  Node node;
  node.Initialize(ninfo);

  GradConstructor grad;
  GradientRegistry::Instance().Lookup(node_def.op(), &grad);
  std::vector<Node*> dy_list;
  dy_list.push_back(&node);

  Graph graph;

  grad(node, dy_list, dy_list, graph);

  return 0;
}
