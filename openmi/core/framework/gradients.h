#ifndef OPENMI_CORE_FRAMEWORK_GRADIENTS_H_
#define OPENMI_CORE_FRAMEWORK_GRADIENTS_H_  

#include "graph.h"

namespace openmi {

class Gradients {
public:
  Gradients() {}

  int gradients(std::vector<Node*>& output_nodes, 
                std::vector<Node*>& input_nodes, 
                std::vector<Node*>& reversed_node_list, 
                Graph* g);

  Node* SumGradients(std::vector<Node*>& node_list, Graph* g);

private:
  // A map form node to a list of gradient contributions from each output node
  std::unordered_map<Node*, std::vector<Node*>* > node_to_grads_list_mapper_;
  // A map from node to the gradient of that node 
  std::unordered_map<Node*, Node*> node_to_output_grad_mapper_;
  // A topological order list from traverse graph given the ouutput nodes that we are taking gradient wrt 
  std::vector<Node*> topo_order_list_;  // from source to sink
}; // class Gradients

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_GRADIENTS_H_ 
