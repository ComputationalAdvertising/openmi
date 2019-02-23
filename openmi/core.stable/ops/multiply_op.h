#ifndef OPENMI_CORE_OPS_MULTIPLY_OP_H_
#define OPENMI_CORE_OPS_MULTIPLY_OP_H_ 

#include "core/framework/op.h"

namespace openmi {

/*!
 * x1:m*k, x2:k*n; y:m*n
 * forward: y = x1 * x2
 * reverse: 
 *    dx1 = dy * x2^T
 *    dx2 = x1^T * dy
 */
class MultiplyOp : public Op {
public:
  MultiplyOp();

  virtual ~MultiplyOp();

  void Compute(Node* node, std::vector<Node*>& input_nodes) override;

  std::vector<Node*>* Gradient(Node* node, std::vector<Node*>& output_nodes, NodeManager* node_manager) override; 

}; // class MultiplyOp

} // namespace openmi
#endif // OPENMI_CORE_OPS_MULTIPLY_OP_H_
