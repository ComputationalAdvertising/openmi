#include "core/framework/node_manager.h"
#include "core/ops/add_op.h"

namespace openmi {

REGISTER_OP(AddOp);
OPENMI_REGISTER_FILE_TAG(add_op);

AddOp::AddOp(): Op("AddOp") {
}

AddOp::~AddOp() {
}

void AddOp::Compute(Node* node, std::vector<Node*>& input_nodes) {    
  LOG(WARNING) << "AddOp::Compute node:" << node->Name();
  CHECK(input_nodes.size() == 2) << " AddOp::Compute input nodes size != 2";
  node->Value() = input_nodes[0]->Value() + input_nodes[1]->Value();

  auto x1 = input_nodes[0]->Data().TensorType<2>();
  auto x2 = input_nodes[1]->Data().TensorType<2>();
  auto y = node->Data().TensorType<2>();
  y = x1 + x2;
}

std::vector<Node*>* AddOp::Gradient(Node* node, std::vector<Node*>& output_nodes, NodeManager* node_manager) {
  LOG(WARNING) << "AddOp::Gradient node:" << node->Name();
  CHECK(output_nodes.size() == 1) << node->DebugString() << " output nodes size != 1. size: " << output_nodes.size();
  std::vector<Node*>* rt = new std::vector<Node*>();
  rt->push_back(output_nodes[0]);
  rt->push_back(output_nodes[0]);
  return rt;
}

} // namespace openmi
