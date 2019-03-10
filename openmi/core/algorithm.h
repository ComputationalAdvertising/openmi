#ifndef OPENMI_CORE_GRAPH_ALGORITHM_H_
#define OPENMI_CORE_GRAPH_ALGORITHM_H_

#include <unordered_set>
#include "graph.h"

using namespace openmi;

namespace openmi {

// A simple algorithm is to do a Depth-First-Search traversal on the give nodes.
extern int TopoOrderList(std::vector<Node*>& node_list, 
                         std::vector<Node*>& topo_order_list,
                         Graph* g);

// post-order DFS
extern int DFS(std::vector<Node*>& node_list, 
               std::unordered_set<Node*> visited, 
               std::vector<Node*>& topo_order_list,
               Graph* g);

} // namespace openmi 
#endif // OPENMI_CORE_GRAPH_ALGORITHM_H_
