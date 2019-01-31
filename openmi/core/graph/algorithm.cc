#include "core/graph/algorithm.h"
#include "base/logging.h"

namespace openmi {
namespace graph {

// post-order DFS 
int DFS(Node* node, std::unordered_set<Node*>& visited, std::vector<Node*>& topo_order_list) {
  if (visited.find(node) != visited.end()) {
    return 0;
  }
  visited.insert(node);
  LOG(INFO) << "----> current_node: " << node->DebugString(); 
  for (Node* n: node->Inputs()) {
    if (n == nullptr) {
      LOG(WARNING) << "node inputs exists null.";
      continue;
    }
    DFS(n, visited, topo_order_list);
  }
  topo_order_list.push_back(node);
  return 0;
}

int TopoOrderList(std::vector<Node*>& node_list, std::vector<Node*>& topo_order_list) {
  topo_order_list.clear();

  std::unordered_set<Node*> visited;
  int count = 0;
  for (size_t i = 0; i < node_list.size(); ++i) {
    Node* node = node_list[i];
    if (DFS(node, visited, topo_order_list) != 0) {
      LOG(ERROR) << "DFS error.";
      return -1;
    }

    std::string topo_linked("");
    for (size_t idx = 0; idx < topo_order_list.size(); ++idx) {
      topo_linked += " --> " + topo_order_list[idx]->Name();
    }
    LOG(INFO) << "[" << count++ << "] " << topo_linked;
  }
  return 0;
}

} // namespace graph
} // namespace openmi
