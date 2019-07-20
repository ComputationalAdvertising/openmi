#include "gradient_registry.h"
#include "session_state.h"
#include "base/logging.h"

int main(int argc, char** argv) {
  proto::NodeDef node_def;
  node_def.set_op("Sigmoid");
  node_def.set_device("CPU");
  NodeInfo ninfo(node_def, -1, NC_UNINITIALIZED, NS_FORWARD);
  Node node;
  node.Initialize(ninfo);

  Node* dy = new Node;
  dy->Initialize(ninfo);
  std::vector<Node*> dy_list;
  dy_list.push_back(dy);

  GradConstructor grad;
  GradientRegistry::Instance().Lookup(node_def.op(), &grad);
  Graph graph;
  SessionState session_state;

  grad(node, dy_list, dy_list, graph, session_state);

  return 0;
}
