#include "core/framework/node_manager.h"
#include "core/ops/sigmoid_grad_op.h"

namespace openmi {

REGISTER_OP(SigmoidGradOp);
OPENMI_REGISTER_FILE_TAG(sigmoid_grad_op);

SigmoidGradOp::SigmoidGradOp(): Op("SigmoidGradOp") {
}

SigmoidGradOp::~SigmoidGradOp() {
}

void SigmoidGradOp::Compute(Node* node, std::vector<Node*>& input_nodes) {
  LOG(WARN) << "SigmoidGradOp::Compute node:" << node->Name();
  LOG(INFO) << "[[0]] tensor Y " << input_nodes[0]->Name() << ", shape: " << input_nodes[0]->Data().Shape().DebugString();
  auto Y = input_nodes[0]->Data().TensorType<2>();
  auto dX = node->Data().TensorType<2>();
  LOG(INFO) << "SigmoidGradOp::Compute Y:\n" << Y;
  
  dX = (1.0f - Y) * Y;
  LOG(INFO) << "[[1]] tensor dX: " << node->Name() << ", shape: " << node->Data().Shape().DebugString();
  LOG(INFO) << "SigmoidGradOp::Compute dX:\n" << dX;
}

std::vector<Node*>* SigmoidGradOp::Gradient(Node* node, std::vector<Node*>& output_nodes, NodeManager* node_manager) {
  LOG(INFO) << "SigmoidGradOp::Gradient node:" << node->Name();
  return nullptr;
}

}
