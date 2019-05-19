#include "graph_utils.h"

namespace openmi {

Node* CreateGradNode(proto::NodeDef& ndef, 
                     Graph& g, const std::string& related_node_name, NodeClass nc, NodeScope ns) {
  Node* related_node = nullptr;
  if (related_node_name != "") {
    related_node = g.FindNode(related_node_name);
    CHECK(related_node != nullptr) << related_node_name << " not in graph.";
  }
  NodeInfo ninfo(ndef, -1, nc, ns);
  Node* grad_node = g.GetOrCreateNode(ninfo, *related_node);
  CHECK(grad_node != nullptr) 
    << "gradient node create failed. "
    << "name[" << ndef.name() << "], op[" << ndef.op() << "]";

  // fill gradient node input 
  for (size_t i = 0; i < ndef.input().size(); ++i) {
    auto input = ndef.input(i);
    Node* n = g.FindNode(input);
    CHECK(n != nullptr) << input << " not in graph. please check it.";
    grad_node->AddInput(input);
  }
  return grad_node;
}

Node* CreateGradNode(const std::string& node_name, const std::string& op, 
                     Graph& g, const std::string& related_node_name, NodeClass nc, NodeScope ns) {
  LOG(DEBUG) << "grad node name:" << node_name << ", op:" << op;
  Node* related_node = g.FindNode(related_node_name);
  CHECK(related_node != nullptr) << related_node_name << " not in graph.";
  proto::NodeDef ndef;
  ndef.set_name(node_name);
  ndef.set_op(op);
  NodeInfo ninfo(ndef, -1, nc, ns);
  Node* grad_node = g.GetOrCreateNode(ninfo, *related_node);
  CHECK(grad_node != nullptr) 
    << "gradient node create failed. name:" << node_name 
    << ", op:" << op;
  return grad_node;
}

}
