#include "gradient_registry.h"
#include "graph_utils.h"
#include "attr_value_utils.h"
#include "base/logging.h"
#include "base/register.h"

namespace openmi {


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
OPENMI_REGISTER_GRADIENT(sigmoid, SigmoidGrad);

// Oneslike gradient
void OneslikeGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g) {
  DLOG(INFO) << "OneslikeGrad ...";
}
OPENMI_REGISTER_GRADIENT(Oneslike, OneslikeGrad);

void ZeroslikeGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g) { 
  DLOG(INFO) << "ZeroslikeGrad ...";
}
OPENMI_REGISTER_GRADIENT(Zeroslike, ZeroslikeGrad);

// MatMul gradient
void MatMulGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g) {
  std::string op("MatMul");
  auto x1_name = node.inputs().at(0);
  auto x2_name = node.inputs().at(1);
  auto y_name = node.def().name();
  auto dy_name = dy_list.at(0)->def().name();

  bool transpose_a_ = false;
  GetAttr<bool>(node.attrs(), "transpose_a", &transpose_a_, ::openmi::AttrValue::kBool);
  bool transpose_b_ = false;
  GetAttr<bool>(node.attrs(), "transpose_b", &transpose_b_, ::openmi::AttrValue::kBool);
  
  std::string dx1_node_name(x1_name + "_GradOp(" + dy_name + " * " + x2_name + "^T)");
  proto::NodeDef ndef1;
  ndef1.set_name(dx1_node_name);
  ndef1.set_op(op);
  
  bool dx1_transpose_a = false, dx1_transpose_b = true;
  if (!transpose_a_) {
    // dA = dY * B^T
    ndef1.add_input(dy_name);
    ndef1.add_input(x2_name);
    if (transpose_b_) {
      LOG(DEBUG) << "update transpose_b";
      dx1_transpose_b = false;
    }
  } else {
    // dA = (dY * B^T)^T = B * dY^T 
    ndef1.add_input(x2_name);
    ndef1.add_input(dy_name);
    if (transpose_b_) {
      LOG(DEBUG) << "update transpose_a";
      dx1_transpose_a = true;
    }
  }
  
  auto attr = ndef1.mutable_attr();
  attr->insert({"transpose_a", *attr_b(dx1_transpose_a)});
  attr->insert({"transpose_b", *attr_b(dx1_transpose_b)});
  Node* dx1 = CreateGradNode(ndef1, g, x1_name, NC_OP, NS_REVERSE);
  DLOG(DEBUG) << "MatMulGrad dx1:\n" << dx1->def().DebugString();
  dx_list.push_back(dx1);

  std::string dx2_node_name(x2_name + "_GradOp(" + x1_name + "^T * " + dy_name + ")");
  proto::NodeDef ndef2;
  ndef2.set_name(dx2_node_name);
  ndef2.set_op(op);
  attr = ndef2.mutable_attr();
  attr->insert({"transpose_a", *attr_b(true)});
  attr->insert({"transpose_b", *attr_b(false)});
  
  bool dx2_transpose_a = true, dx2_transpose_b = false;
  if (!transpose_b_) {
    // dB = A^T * dY 
    ndef2.add_input(x1_name);
    ndef2.add_input(dy_name);
    if (transpose_a_) {
      dx2_transpose_a = false;
    }
  } else {
    // dB = dY^T * dA 
    ndef2.add_input(dy_name);
    ndef2.add_input(x1_name);
    if (transpose_a_) {
      dx2_transpose_b = true;
    }
  }
  Node* dx2 = CreateGradNode(ndef2, g, x2_name, NC_OP, NS_REVERSE);
  DLOG(DEBUG) << "MatMulGrad dx2:\n" << dx2->def().DebugString();
  dx_list.push_back(dx2);
}
OPENMI_REGISTER_GRADIENT(MatMul, MatMulGrad);

// Add gradient
void AddGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g) {
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

// Softmax gradient 
void SoftmaxGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g) {
  // logits
  auto x1_name = node.inputs().at(0);
  auto dy_name = dy_list.at(0)->def().name();
  auto dy_input_name = dy_list.at(0)->inputs().at(0);
  auto dy_input_node = g.FindNode(dy_input_name);
  // TODO 
  DLOG(INFO) << __FUNCTION__ << ". dy_name: " << dy_name << ", dy_input_name:" << dy_input_name;
}
OPENMI_REGISTER_GRADIENT(Softmax, SoftmaxGrad);


// Softmax Cross Entropy with Logits
void SoftmaxCrossEntropyWithLogitsGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g) {
  // labels 
  auto x1_name = node.inputs().at(0);
  // logits
  auto x2_name = node.inputs().at(1); 
  // softmax_cross_entropy 
  auto y_name = node.def().name();
  // oneslike(1)
  auto dy_name = dy_list.at(0)->def().name();

  std::string op("softmax_cross_entropy_with_logits_grad");
  std::string dx2_node_name(x2_name + "_GradOp(" + dy_name + ")");
  // TODO
}
OPENMI_REGISTER_GRADIENT(softmax_cross_entropy_with_logits, SoftmaxCrossEntropyWithLogitsGrad);

void SigmoidCrossEntropyWithLogitsGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g) {
  DLOG(INFO) << __FUNCTION__;
  // labels
  auto x1_name = node.inputs().at(0);
  // logits
  auto x2_name = node.inputs().at(1);
  // loss
  auto y_name = node.def().name();
  // Oneslike(loss)
  auto dy_name = dy_list.at(0)->def().name();

  std::string dx1_node_name(x1_name + "_GradOp(" + dy_name + ")");
  proto::NodeDef ndef1;
  ndef1.set_name(dx1_node_name);
  ndef1.set_op("Oneslike");
  Node* dx1 = CreateGradNode(ndef1, g, x1_name, NC_OP, NS_REVERSE);
  dx_list.push_back(dx1);

  std::string dx2_node_name(x2_name + "_GradOp(" + dy_name + ")");
  node.AddOutput(dx2_node_name);
  DLOG(INFO) << "node.name:" << y_name << ", node.output(0):" << node.outputs().at(0);
  proto::NodeDef ndef2;
  ndef2.set_name(dx2_node_name);
  //ndef2.set_op("sigmoid_cross_entropy_with_logits_grad");
  ndef2.set_op("no_gradient");
  ndef2.add_input(x1_name);
  ndef2.add_input(x2_name);
  ndef2.add_input(dy_name);
  ndef2.add_input(y_name);
  Node* dx2 = CreateGradNode(ndef2, g, x2_name, NC_OP, NS_REVERSE);
  dx_list.push_back(dx2);
}
OPENMI_REGISTER_GRADIENT(sigmoid_cross_entropy_with_logits, SigmoidCrossEntropyWithLogitsGrad);

OPENMI_REGISTER_GRADIENT(Variable, OneslikeGrad);
OPENMI_REGISTER_GRADIENT(Placeholder, OneslikeGrad);

////////////////////////////// 
OPENMI_REGISTER_FILE_TAG(math_grad_constructor);

} // namespace openmi
