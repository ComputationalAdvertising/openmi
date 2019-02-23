#include "core/framework/node_manager.h"
#include "core/ops/placeholder_op.h"

namespace openmi {

REGISTER_OP(PlaceholderOp);
OPENMI_REGISTER_FILE_TAG(placeholder_op);

PlaceholderOp::PlaceholderOp(): Op("PlaceholderOp") {
}

PlaceholderOp::~PlaceholderOp() {
}

void PlaceholderOp::Compute(Node* node, std::vector<Node*>& input_nodes) {    
  LOG(WARNING) << "PlaceholderOp::Compute node:" << node->Name();
}

std::vector<Node*>* PlaceholderOp::Gradient(Node* node, std::vector<Node*>& output_nodes, NodeManager* node_manager) {
  LOG(WARNING) << "PlaceholderOp::Gradient node:" << node->Name();
  CHECK(output_nodes.size() == 1) << node->DebugString() << " output nodes size != 1. size: " << output_nodes.size();
  std::vector<Node*>* rt = new std::vector<Node*>();
  rt->push_back(output_nodes[0]);
  rt->push_back(output_nodes[0]);
  return rt;
}

} // namespace openmi
