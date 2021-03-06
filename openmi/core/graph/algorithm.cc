#include "algorithm.h"
#include "base/logging.h"

namespace openmi {

// post-order DFS 
int DFS(Node* node, std::unordered_set<Node*>& visited, std::vector<Node*>& topo_order_list, Graph* g) {
  if (visited.find(node) != visited.end()) {
    return 0;
  }
  visited.insert(node);
  DLOG(INFO) << "----> current_node: " << node->def().name(); 
  for (std::string& input: node->inputs()) {
    auto input_node = g->FindNode(input);
    if (input_node == nullptr) {
      LOG(ERROR) << "node '" << input << "' input null.";
      continue;
    }
    DFS(input_node, visited, topo_order_list, g);
  }

  topo_order_list.push_back(node);
  return 0;
}

int TopoOrderList(std::vector<Node*>& node_list, std::vector<Node*>& topo_order_list, Graph* g) {
  topo_order_list.clear();

  std::unordered_set<Node*> visited;
  for (size_t i = 0; i < node_list.size(); ++i) {
    Node* node = node_list[i];

    if (DFS(node, visited, topo_order_list, g) != 0) {
      LOG(ERROR) << "DFS error.";
      return -1;
    }
  }
  return 0;
}

} // namespace openmi
