#include "core/framework/node_manager.h"
#include "core/ops/multiply_op.h"

namespace openmi {

REGISTER_OP(MultiplyOp);
OPENMI_REGISTER_FILE_TAG(multiply_op);

MultiplyOp::MultiplyOp(): Op("MultiplyOp") {
}

MultiplyOp::~MultiplyOp() {
}

void MultiplyOp::Compute(Node* node, std::vector<Node*>& input_nodes) {
  LOG(WARNING) << "MultiplyOp::Compute node: " << node->Name();
  CHECK(input_nodes.size() == 2) << "size of input nodes not equal to 2. size: " << input_nodes.size();
  node->Value() = input_nodes[0]->Value() * input_nodes[1]->Value();
  LOG(INFO) << "[[0]] tensor name: " << input_nodes[0]->Name() << ", shape: " << input_nodes[0]->Data().Shape().DebugString();
  LOG(INFO) << "[[1]] tensor name: " << input_nodes[1]->Name() << ", shape: " << input_nodes[1]->Data().Shape().DebugString();
  auto x1 = input_nodes[0]->Data().TensorType<2>();
  auto x2 = input_nodes[1]->Data().TensorType<2>();
  auto y = node->Data().TensorType<2>();
  y = x1 * x2;

  LOG(INFO) << "\nx1:\n" << x1 << ",\nx2:\n" << x2 << ",\ny:\n" << y;
  
  if (node->Name() == "(w*SigmoidGradOp(y))") {
    LOG(INFO) << "\n============== (w*SigmoidGradOp(y)) ==================\n";
    LOG(INFO) << input_nodes[0]->Name() << ", value:\n" << x1;
    LOG(INFO) << input_nodes[1]->Name() << ", value:\n" << x2;
    LOG(INFO) << node->Name() << ", y:\n" << y;
    LOG(INFO) << "\n============== (w*SigmoidGradOp(y)) done ==================\n";
  }
}

std::vector<Node*>* MultiplyOp::Gradient(Node* node, std::vector<Node*>& output_nodes, NodeManager* node_manager) {
  LOG(WARNING) << "MultiplyOp::Gradient node: " << node->Name();
  std::vector<Node*>* rt = new std::vector<Node*>();

  // dx1 = dy * x2^T = grad (output_nodes[0]) * node.inputs[1]^T
  std::string node0_name("(" + node->Inputs()[1]->Name() + "*" + output_nodes[0]->Name() + ")");
  NodePtr node0 = node_manager->GetOrCreate(node0_name, 0, node->Inputs()[0]->Data().Shape(), this->Name(), NT_OP, NCT_REVERSE);
  if (node0 == nullptr) {
    LOG(ERROR) << "create new node failed. name: " << node0_name;
    return NULL;
  }

  // new node 
  if (node0->Inputs().size() == 0) {
    auto x2 = node->Inputs()[1];
    auto dy = output_nodes[0];
    node0->AddInput(dy);
    node0->AddInput(x2);
  }

  rt->push_back(node0.get());

  // node.inputs[0] * grad (output_nodes[0])
  std::string node1_name("(" + node->Inputs()[0]->Name() + "*" + output_nodes[0]->Name() + ")");
  NodePtr node1 = node_manager->GetOrCreate(node1_name, 0, node->Inputs()[1]->Data().Shape(), this->Name(), NT_OP, NCT_REVERSE);
  if (node1 == nullptr) {
    LOG(ERROR) << "create new node failed. name: " << node1_name;
    return NULL;
  }
  if (node1->Inputs().size() == 0) {
    auto dy = output_nodes[0];
    auto x1 = node->Inputs()[0];
    node1->AddInput(x1);
    node1->AddInput(dy);
  }

  rt->push_back(node1.get());

  return rt;
}

} // namespace openmi
