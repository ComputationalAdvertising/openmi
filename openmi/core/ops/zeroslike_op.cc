#include "core/ops/zeroslike_op.h"
#include "core/framework/node_manager.h"

namespace openmi {

REGISTER_OP(ZeroslikeOp);
OPENMI_REGISTER_FILE_TAG(zeroslike_op);

ZeroslikeOp::ZeroslikeOp(): Op("ZeroslikeOp") {
}

ZeroslikeOp::~ZeroslikeOp() {
}

void ZeroslikeOp::Compute(Node* node, std::vector<Node*>& input_nodes) {
  LOG(WARNING) << "---------- ZeroslikeOp::Compute node:" << node->Name();
  // TODO tensor.zeros(input_nodes[0].shape) to node->SetValue
  int value = 0;
  node->SetValue(value);
  auto tensor = node->Data().TensorType<2>();
  tensor.setZero();
  LOG(WARNING) << "ZeroslikeOp::Compute tensor:\n" << tensor;
}

std::vector<Node*>* ZeroslikeOp::Gradient(Node* node, std::vector<Node*>& output_nodes, NodeManager* node_manager) {
  LOG(WARNING) << "ZeroslikeOp::Gradient node:" << node->Name();
  std::string n_name(this->Name() + "(" + node->Name() + ")");
  //NodePtr n = node_manager->GetOrCreate(n_name, 0, this->Name(), NT_SOURCE, NCT_REVERSE);
  NodePtr n = node_manager->GetOrCreate(n_name, 0, node->Data().Shape(), this->Name(), NT_SOURCE, NCT_REVERSE);
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
