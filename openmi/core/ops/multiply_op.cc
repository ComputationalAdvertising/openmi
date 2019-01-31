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
  
  auto x1 = input_nodes[0]->Data().TensorType<2>();
  auto x2 = input_nodes[1]->Data().TensorType<2>();
  auto y = node->Data().TensorType<2>();
  y = x1 * x2;
  
}

std::vector<Node*>* MultiplyOp::Gradient(Node* node, std::vector<Node*>& output_nodes, NodeManager* node_manager) {
  LOG(WARNING) << "MultiplyOp::Gradient node: " << node->Name();
  std::vector<Node*>* rt = new std::vector<Node*>();

  // node.inputs[1] * grad (output_nodes[0])
  std::string node0_name("(" + node->Inputs()[1]->Name() + "*" + output_nodes[0]->Name() + ")");
  NodePtr node0 = node_manager->GetOrCreate(node0_name, 0, this->Name(), 2);
  if (node0 == nullptr) {
    LOG(ERROR) << "create new node failed. name: " << node0_name;
    return NULL;
  }

  if (node0->Inputs().size() == 0) {   // new node
    node0->AddInput(node->Inputs()[1]);
    node0->AddInput(output_nodes[0]);
  }

  rt->push_back(node0.get());

  // node.inputs[0] * grad (output_nodes[0])
  std::string node1_name("(" + node->Inputs()[0]->Name() + "*" + output_nodes[0]->Name() + ")");
  NodePtr node1 = node_manager->GetOrCreate(node1_name, 0, this->Name(), 2);
  if (node1 == nullptr) {
    LOG(ERROR) << "create new node failed. name: " << node1_name;
    return NULL;
  }
  if (node1->Inputs().size() == 0) {   // new node
    node1->AddInput(node->Inputs()[0]);
    node1->AddInput(output_nodes[0]);
  }

  rt->push_back(node1.get());

  return rt;
}

} // namespace openmi
