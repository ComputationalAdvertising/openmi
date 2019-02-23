#ifndef OPENMI_CORE_GRAPH_ALGORITHM_H_
#define OPENMI_CORE_GRAPH_ALGORITHM_H_

#include <unordered_set>
#include "core/framework/node.h"

using namespace openmi;

namespace openmi {
namespace graph {

// A simple algorithm is to do a Depth-First-Search traversal on the give nodes.
extern int TopoOrderList(std::vector<Node*>& node_list, std::vector<Node*>& topo_order_list);

// post-order DFS
extern int DFS(std::vector<Node*>& node_list, std::unordered_set<Node*> visited, std::vector<Node*>& topo_order_list);

} // namespace graph
} // namespace openmi 
#endif // OPENMI_CORE_GRAPH_ALGORITHM_H_
