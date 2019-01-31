#ifndef OPENMI_CORE_OPS_MULTIPLY_OP_H_
#define OPENMI_CORE_OPS_MULTIPLY_OP_H_ 

#include "core/framework/op.h"

namespace openmi {

class MultiplyOp : public Op {
public:
  MultiplyOp();

  virtual ~MultiplyOp();

  void Compute(Node* node, std::vector<Node*>& input_nodes) override;

  std::vector<Node*>* Gradient(Node* node, std::vector<Node*>& output_nodes, NodeManager* node_manager) override; 

}; // class MultiplyOp

} // namespace openmi
#endif // OPENMI_CORE_OPS_MULTIPLY_OP_H_
