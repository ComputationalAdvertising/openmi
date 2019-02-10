#include "core/framework/op.h"
#include "base/logging.h"

namespace openmi {

Op::Op(std::string op_name): op_name_(op_name) {
}

Op::~Op() {
}

void Op::Compute(Node* node, std::vector<Node*>& input_nodes) {
  LOG(INFO) << "Op::Compute";
} 

void Op::Compute(Node* node, std::vector<Node*>& input_nodes, OpKernelContext* ctx) {
  LOG(INFO) << "Op::Compute with OpKernelContext";
}

std::vector<Node*>* Op::Gradient(Node* node, std::vector<Node*>& output_nodes, NodeManager* node_manager) {
  LOG(WARNING) << "Op::Gradient. return nullptr";
  return nullptr;
} 

}// namespace openmi
