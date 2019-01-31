#ifndef OPENMI_CORE_OPS_PLACEHOLDER_H_
#define OPENMI_CORE_OPS_PLACEHOLDER_H_ 

#include "core/framework/op.h"

namespace openmi {

class PlaceholderOp: public Op {
public:
  PlaceholderOp();

  virtual ~PlaceholderOp();

  void Compute(Node* node, std::vector<Node*>& input_nodes) override;

  std::vector<Node*>* Gradient(Node* node, std::vector<Node*>& output_nodes, NodeManager* node_manager) override; 

}; // class PlaceholderOp

} // namespace openmi  
#endif // OPENMI_CORE_OPS_PLACEHOLDER_H_
