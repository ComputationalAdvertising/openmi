#ifndef OPENMI_GRAPH_GRPAH_H_ 
#define OPENMI_GRAPH_GRPAH_H_ 

#include <string>
#include <vector>

#include "core/framework/node.h"

using namespace openmi;

namespace openmi {
namespace graph {

class Graph {
private:
  std::vector<NodePtr> nodes_;
  uint64_t num_nodes_ = 0;
}; // class Graph

}
}
#endif // OPENMI_GRAPH_GRPAH_H_
