#include "core/framework/node_manager.h"
#include "core/ops/oneslike_op.h"

namespace openmi {

REGISTER_OP(OneslikeOp);
OPENMI_REGISTER_FILE_TAG(oneslike_op);

OneslikeOp::OneslikeOp(): Op("OneslikeOp") {
}

OneslikeOp::~OneslikeOp() {
}

void OneslikeOp::Compute(Node* node, std::vector<Node*>& input_nodes) {
  LOG(WARNING) << "OneslikeOp::Compute node:" << node->Name();
  node->Value() = 1;
  
  auto tensor = node->Data().TensorType<2>();
  tensor.setConstant(1);
}

std::vector<Node*>* OneslikeOp::Gradient(Node* node, std::vector<Node*>& output_nodes, NodeManager* node_manager) {
  LOG(WARNING) << "OneslikeOp::Gradient node:" << node->Name();
  std::string n_name(this->Name() + "(" + node->Name() + ")");
  //NodePtr n = node_manager->GetOrCreate(n_name, 0, "ZeroslikeOp", NT_SOURCE, NCT_REVERSE);
  NodePtr n = node_manager->GetOrCreate(n_name, 0, node->Data().Shape(), "ZeroslikeOp", NT_SOURCE, NCT_REVERSE);
  if (n == nullptr) {
    LOG(ERROR) << "create new node failed. name: " << n_name << ", op_name: " << this->Name();
    return NULL;
  }
  n->AddInput(node->Inputs()[0]);
  std::vector<Node*>* rt = new std::vector<Node*>();
  rt->push_back(n.get());
  return rt;
}

} // namespace openmi
