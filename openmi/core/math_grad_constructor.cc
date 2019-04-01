#include "gradient_registry.h"
#include "attr_value_utils.h"
#include "base/logging.h"
#include "base/register.h"

namespace openmi {

Node* CreateGradNode(const std::string& node_name, const std::string& op, 
                     Graph& g, const std::string& related_node_name, 
                     NodeClass nc = NC_OP, NodeScope ns = NS_REVERSE) {
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

// ReduceSum gradient
void ReduceSumGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g) {
  auto x_name = node.inputs().at(0);
  auto dy_name = dy_list.at(0)->def().name();
  std::string op("ReduceSumGrad");
  std::string dx_node_name(x_name + "_GradOp(" + dy_name + ")");
  Node* dx = CreateGradNode(dx_node_name, op, g, node.inputs().at(0));
  dx->AddInput(dy_name);
  dx_list.push_back(dx);
}
OPENMI_REGISTER_GRADIENT(ReduceSum, ReduceSumGrad);

// Sigmoid gradient
void SigmoidGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g) {
  auto x_name = node.inputs().at(0);
  auto y_name = node.def().name();
  auto dy_name = dy_list.at(0)->def().name();
  std::string op("SigmoidGrad");  // TODO -> SigmoidGrad
  std::string dx_node_name(x_name + "_GradOp(" + dy_name + "*" + y_name + "*(1-" + y_name + "))");
  Node* dx = CreateGradNode(dx_node_name, op, g, x_name, NC_OP, NS_REVERSE);
  dx->AddInput(y_name);
  dx->AddInput(dy_name);
  dx_list.push_back(dx);
}
OPENMI_REGISTER_GRADIENT(Sigmoid, SigmoidGrad);

// Oneslike gradient
void OneslikeGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g) {
  LOG(DEBUG) << "OneslikeGrad ...";
}
OPENMI_REGISTER_GRADIENT(Oneslike, OneslikeGrad);

void ZeroslikeGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g) { 
  LOG(DEBUG) << "OneslikeGrad ...";
}
OPENMI_REGISTER_GRADIENT(Zeroslike, ZeroslikeGrad);

// MatMul gradient
void MatMulGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g) {
  LOG(INFO) << "MatMulGrad ...";
  std::string op("MatMul");
  auto x1_name = node.inputs().at(0);
  auto x2_name = node.inputs().at(1);
  auto y_name = node.def().name();
  auto dy_name = dy_list.at(0)->def().name();
  std::string dx1_node_name(x1_name + "_GradOp(" + dy_name + " * " + x2_name + "^T)");
  Node* dx1 = CreateGradNode(dx1_node_name, op, g, x1_name, NC_OP, NS_REVERSE);
  dx1->AddInput(dy_name);
  dx1->AddInput(x2_name);
  auto attr = const_cast<proto::NodeDef&>(dx1->def()).mutable_attr();
  attr->insert({"transpose_a", *attr_b(false)});
  attr->insert({"transpose_b", *attr_b(true)});
  dx_list.push_back(dx1);

  std::string dx2_node_name(x2_name + "_GradOp(" + x1_name + "^T * " + dy_name + ")");
  Node* dx2 = CreateGradNode(dx2_node_name, op, g, x2_name, NC_OP, NS_REVERSE);
  dx2->AddInput(node.inputs()[0]);
  dx2->AddInput(dy_list[0]->def().name());
  attr = const_cast<proto::NodeDef&>(dx2->def()).mutable_attr();
  attr->insert({"transpose_a", *attr_b(true)});
  attr->insert({"transpose_b", *attr_b(false)});
  dx_list.push_back(dx2);
}
OPENMI_REGISTER_GRADIENT(MatMul, MatMulGrad);

// Add gradient
void AddGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g) {
  LOG(INFO) << "AddGrad ...";
  auto x1_name = node.inputs().at(0);
  auto x2_name = node.inputs().at(1);
  auto y_name = node.def().name();
  auto dy_name = dy_list.at(0)->def().name();
  
  std::string op("AddGrad"); // TODO Relu -> UnaryMap
  std::string dx1_node_name(x1_name + "_GradOp(" + dy_name + ")");
  Node* dx1 = CreateGradNode(dx1_node_name, op, g, x1_name, NC_OP, NS_REVERSE);
  dx1->AddInput(dy_name);
  dx_list.push_back(dx1);
  
  std::string dx2_node_name(x2_name + "_GradOp(" + dy_name + ")");
  Node* dx2 = CreateGradNode(dx2_node_name, op, g, x2_name, NC_OP, NS_REVERSE);
  dx2->AddInput(dy_name);
  dx_list.push_back(dx2);
}
OPENMI_REGISTER_GRADIENT(Add, AddGrad);

OPENMI_REGISTER_GRADIENT(Variable, OneslikeGrad);
OPENMI_REGISTER_GRADIENT(Placeholder, OneslikeGrad);

////////////////////////////// 
OPENMI_REGISTER_FILE_TAG(math_grad_constructor);

} // namespace openmi
