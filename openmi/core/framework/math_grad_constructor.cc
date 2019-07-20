#include "gradient_registry.h"
#include "graph_utils.h"
#include "attr_value_utils.h"
#include "base/logging.h"
#include "base/register.h"

namespace openmi {

// Add gradient
void AddGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g, SessionState& session_state) {
  auto x1_name = node.inputs().at(0);
  auto x2_name = node.inputs().at(1);
  auto y_name = node.def().name();
  auto dy_name = dy_list.at(0)->def().name();
  
  std::string op("Assign");
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

// Multiply gradient
void MultiplyGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g, SessionState& session_state) {
  CHECK(node.inputs().size() == 2) << "Multiply op kernel input size not 2. but:" << node.inputs().size();
  auto x1_name = node.inputs().at(0);
  auto x2_name = node.inputs().at(1);
  auto y_name = node.def().name();
  auto dy_name = dy_list.at(0)->def().name();

  std::string op("Multiply");
  std::string dx1_node_name(x1_name + "_GradOp(" + dy_name + ")");
  Node* dx1 = CreateGradNode(dx1_node_name, op, g, x1_name, NC_OP, NS_REVERSE);
  dx1->AddInput(dy_name);
  dx1->AddInput(x2_name);
  dx_list.push_back(dx1);
  
  std::string dx2_node_name(x2_name + "_GradOp(" + dy_name + ")");
  Node* dx2 = CreateGradNode(dx2_node_name, op, g, x2_name, NC_OP, NS_REVERSE);
  dx2->AddInput(dy_name);
  dx2->AddInput(x1_name);
  dx_list.push_back(dx2);
}
OPENMI_REGISTER_GRADIENT(Multiply, MultiplyGrad);

// Concat gradient
void ConcatGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g, SessionState& session_state) {
  auto dy_name = dy_list.at(0)->def().name();
  size_t input_size = node.inputs().size();

  std::string op("Slice");
  proto::NodeDef ndef;
  ndef.set_name(op + "_GradOp(" + dy_name + ")");
  ndef.set_op(op);
  ndef.set_device("CPU");
  auto attr = ndef.mutable_attr();
  attr->insert({"input_size", *attr_i((int)input_size)});
  Node* concat_grad_node = CreateGradNode(ndef, g);
  CHECK(concat_grad_node != nullptr) << __FUNCTION__ << " create grad node failed.";
  concat_grad_node->AddInput(dy_name);

  for (size_t i = 0; i < input_size; ++i) {
    auto xi_name = node.inputs().at(i);
    std::string dxi_name(xi_name + "_GradOp(" + dy_name + ")");
    proto::NodeDef dxi_ndef;
    dxi_ndef.set_name(dxi_name);
    dxi_ndef.set_op("nothing");
    Node* dxi = CreateGradNode(dxi_name, dxi_ndef.op(), g, xi_name);
    dx_list.push_back(dxi);
    
    concat_grad_node->AddInput(xi_name);
    concat_grad_node->AddOutput(dxi_name);
  }
}
OPENMI_REGISTER_GRADIENT(Concat, ConcatGrad);

// ReduceSum gradient
void ReduceSumGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g, SessionState& session_state) {
  auto x_name = node.inputs().at(0);
  auto dy_name = dy_list.at(0)->def().name();
  std::string op("ReduceSumGrad");
  std::string dx_node_name(x_name + "_GradOp(" + dy_name + ")");
  Node* dx = CreateGradNode(dx_node_name, op, g, node.inputs().at(0));
  dx->AddInput(dy_name);
  dx_list.push_back(dx);
}
OPENMI_REGISTER_GRADIENT(ReduceSum, ReduceSumGrad);

// MatMul gradient
void MatMulGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g, SessionState& session_state) {
  std::string op("MatMul");
  auto x1_name = node.inputs().at(0);
  auto x2_name = node.inputs().at(1);
  auto y_name = node.def().name();
  auto dy_name = dy_list.at(0)->def().name();

  bool transpose_a_ = false;
  GetAttr(node.attrs(), "transpose_a", &transpose_a_);
  bool transpose_b_ = false;
  GetAttr(node.attrs(), "transpose_b", &transpose_b_);
  
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
      dx1_transpose_b = false;
    }
  } else {
    // dA = (dY * B^T)^T = B * dY^T 
    ndef1.add_input(x2_name);
    ndef1.add_input(dy_name);
    if (transpose_b_) {
      dx1_transpose_a = true;
    }
  }
  
  auto attr = ndef1.mutable_attr();
  attr->insert({"transpose_a", *attr_b(dx1_transpose_a)});
  attr->insert({"transpose_b", *attr_b(dx1_transpose_b)});
  Node* dx1 = CreateGradNode(ndef1, g, x1_name, NC_OP, NS_REVERSE);
  DLOG(INFO) << __FUNCTION__ << " dx1:\n" << dx1->def().DebugString();
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
  DLOG(INFO) << __FUNCTION__ << "dx2:\n" << dx2->def().DebugString();
  dx_list.push_back(dx2);
}
OPENMI_REGISTER_GRADIENT(MatMul, MatMulGrad);

// SegmentSum
void SegmentSumGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g, SessionState& session_state) {
  auto x_name = node.inputs().at(0);
  auto offset_name = node.inputs().at(1);
  auto y = node.def().name();
  auto dy_name = dy_list.at(0)->def().name();
  std::string op("SegmentSumGrad");
  std::string dx_name(x_name + "_grad(" + dy_name + ")");
  proto::NodeDef dx_ndef;
  dx_ndef.set_name(dx_name);
  dx_ndef.set_op(op);
  auto attr = dx_ndef.mutable_attr();
  attr->insert({"is_backward", *attr_b(true)});
  Node* dx = CreateGradNode(dx_ndef, g, x_name, NC_OP, NS_REVERSE);
  dx->AddInput(dy_name);
  dx->AddInput(offset_name);
  dx_list.push_back(dx);
  
  std::string doffset_name(offset_name + "_grad(" + dy_name + ")");
  Node* doffset = CreateGradNode(doffset_name, "Oneslike", g, offset_name, NC_OP, NS_REVERSE);
  dx_list.push_back(doffset);
}
OPENMI_REGISTER_GRADIENT(SegmentSum, SegmentSumGrad);

// SegmentMean
void SegmentMeanGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g, SessionState& session_state) {
  auto x_name = node.inputs().at(0);
  auto offset_name = node.inputs().at(1);
  auto y = node.def().name();
  auto dy_name = dy_list.at(0)->def().name();
  std::string op("SegmentMeanGrad");
  std::string dx_name(x_name + "_grad(" + dy_name + ")");
  Node* dx = CreateGradNode(dx_name, op, g, x_name, NC_OP, NS_REVERSE);
  dx->AddInput(dy_name);
  dx->AddInput(offset_name);
  dx_list.push_back(dx);
}
OPENMI_REGISTER_GRADIENT(SegmentMean, SegmentMeanGrad);

// Sigmoid gradient
void SigmoidGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g, SessionState& session_state) {
  auto x_name = node.inputs().at(0);
  auto y_name = node.def().name();
  auto dy_name = dy_list.at(0)->def().name();
  std::string op("SigmoidGrad");
  std::string dx_node_name(x_name + "_GradOp(" + dy_name + "*" + y_name + "*(1-" + y_name + "))");
  Node* dx = CreateGradNode(dx_node_name, op, g, x_name, NC_OP, NS_REVERSE);
  dx->AddInput(y_name);
  dx->AddInput(dy_name);
  dx_list.push_back(dx);
}
OPENMI_REGISTER_GRADIENT(Sigmoid, SigmoidGrad);
OPENMI_REGISTER_GRADIENT(sigmoid, SigmoidGrad);

// Oneslike gradient
void OneslikeGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g, SessionState& session_state) {
  DLOG(INFO) << "OneslikeGrad ...";
}
OPENMI_REGISTER_GRADIENT(Oneslike, OneslikeGrad);

void ZeroslikeGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g, SessionState& session_state) { 
  DLOG(INFO) << "ZeroslikeGrad ...";
}
OPENMI_REGISTER_GRADIENT(Zeroslike, ZeroslikeGrad);

// Softmax gradient 
void SoftmaxGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g, SessionState& session_state) {
  // logits
  auto x1_name = node.inputs().at(0);
  auto dy_name = dy_list.at(0)->def().name();
  auto dy_input_name = dy_list.at(0)->inputs().at(0);
  //auto dy_input_node = g.FindNode(dy_input_name);
  // TODO gradient impl
  DLOG(INFO) << __FUNCTION__ << ". dy_name: " << dy_name << ", dy_input_name:" << dy_input_name;
}
OPENMI_REGISTER_GRADIENT(Softmax, SoftmaxGrad);

// Softmax Cross Entropy with Logits
void SoftmaxCrossEntropyWithLogitsGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g, SessionState& session_state) {
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
  // TODO gradient impl
}
OPENMI_REGISTER_GRADIENT(softmax_cross_entropy_with_logits, SoftmaxCrossEntropyWithLogitsGrad);

void SigmoidCrossEntropyWithLogitsGrad(Node& node, std::vector<Node*>& dy_list, std::vector<Node*>& dx_list, Graph& g, SessionState& session_state) {
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
  ndef2.set_op("nothing");
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
