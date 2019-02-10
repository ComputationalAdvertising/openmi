#include "core/framework/node_manager.h"
#include "core/ops/sigmoid_op.h"

namespace openmi {

REGISTER_OP(SigmoidOp);
OPENMI_REGISTER_FILE_TAG(sigmoid_op);

SigmoidOp::SigmoidOp(): Op("SigmoidOp") {
}

SigmoidOp::~SigmoidOp() {
}

void SigmoidOp::Compute(Node* node, std::vector<Node*>& input_nodes) {
  LOG(WARNING) << "SigmoidOp::Compute node:" << node->Name();
  auto X = input_nodes[0]->Data().TensorType<2>();
  auto Y = node->Data().TensorType<2>();
  auto norm_fn = [](float x, float eps=16.0f) -> float {
    return (x < -eps) ? -eps : ((x > eps) ? eps : x);
  };

  Y = ((-X.unaryExpr(norm_fn)).exp() + 1.0f).inverse();
  LOG(DEBUG) << "SigmoidOp::Compute Y:\n" << Y;
}

// dX = (1 - y) * y
std::vector<Node*>* SigmoidOp::Gradient(Node* node, std::vector<Node*>& output_nodes, NodeManager* node_manager) {
  CHECK(output_nodes.size() == 1) << node->DebugString() << " output nodes size != 1, but size:" << output_nodes.size();
  std::vector<Node*>* rt = new std::vector<Node*>();
  std::string node_name("SigmoidGradOp(" + node->Name() + ")");
  NodePtr n = node_manager->GetOrCreate(node_name, 0, node->Data().Shape(), "SigmoidGradOp", NT_OP, NCT_REVERSE);
  if (n == nullptr) {
    LOG(ERROR) << "create new node failed. name: " << node_name;
    return NULL;
  }

  if (n->Inputs().size() == 0) {
    n->AddInput(node);
  }

  rt->push_back(n.get());
  return rt;
}

}
