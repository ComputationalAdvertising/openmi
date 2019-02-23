#ifndef OPENMI_CORE_FRAMEWORK_EXECUTOR_H_
#define OPENMI_CORE_FRAMEWORK_EXECUTOR_H_ 

#include "core/graph/algorithm.h"

namespace openmi {

// Compute values for a given subset of nodes in a compuatation graph
class Executor {
public:
  explicit Executor(std::vector<Node*> computed_node_list);
  
  ~Executor();

  int Run();

private:
  // List of nodes whose need to be computed
  std::vector<Node*> nodes_to_be_computed_;
  // Topological order list of nodes whose need to be computed 
  std::vector<Node*> topo_order_list_;

}; // class Executor

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_EXECUTOR_H_ 
