#include "core/framework/executor.h"
#include "core/framework/gradients.h"
#include "core/framework/node_manager.h"
#include "base/protobuf_op.h"
#include "base/timer.h"

using namespace openmi;

NodeManagerPtr ParseNodeList(const char* file) {
  pb::NodeList nodes_pb;
  if (ProtobufOp::LoadObjectFromPbFile<pb::NodeList>(file, &nodes_pb) != 0) {
    LOG(ERROR) << "load node list pb file failed.";
    return nullptr;
  }

  NodeManagerPtr node_mgr2 = std::make_shared<NodeManager>(nodes_pb);
  return node_mgr2;
}

void exec_simple_calc();
void exec_sigmoid_calc();

int main(int argc, char** argv) {
  exec_sigmoid_calc();
  //exec_simple_calc();
  return 0;
}

void exec_sigmoid_calc() {
  const char* pbfile = "./unittest/conf/logistic_regression.graph";
  NodeManagerPtr node_mgr = ParseNodeList(pbfile);

  std::vector<Node*> output_nodes;
  NodePtr y = node_mgr->Get("y");
  output_nodes.push_back(y.get()); 

  std::vector<Node*> input_nodes;
  input_nodes.push_back(node_mgr->Get("w").get());
  input_nodes.push_back(node_mgr->Get("x").get()); 
  input_nodes.push_back(node_mgr->Get("b").get()); 
  
  LOG(INFO) << "total nodes number of forward: " << node_mgr->TotalNodes().size();

  Gradients grad;
  std::vector<Node*> rt;
  int result = grad.gradients(output_nodes, input_nodes, rt, node_mgr.get());
  auto dw = rt[0];
  auto dx = rt[1];
  auto db = rt[2];
  
  LOG(INFO) << "--------------- dx:\n" << dx->DebugString();
  LOG(INFO) << "--------------- dx:\n" << dx->Data().TensorType<2>();

  LOG(INFO) << "total nodes number of forward and reverse: " << node_mgr->TotalNodes().size() 
            << ", reverse node size: " << node_mgr->ReversedNodes().size();; 

  NodePtr w = node_mgr->Get("w");
  NodePtr x = node_mgr->Get("x");
  NodePtr b = node_mgr->Get("b");
  // TODO init source node value 
  w->Data().TensorType<2>().setConstant(1);
  x->Data().TensorType<2>().setConstant(3);
  b->Data().TensorType<2>().setConstant(1);

  // executor 
  output_nodes.push_back(dw);
  output_nodes.push_back(dx);
  output_nodes.push_back(db);
  Executor* executor = new Executor(output_nodes);
  LOG(INFO) << "\n============\nexecute ....\n===========\n";
  executor->Run();

  // result 
  auto y_val = node_mgr->Get("y")->Data().TensorType<2>();
  LOG(INFO) << "y_val:\n" << y_val;
  LOG(INFO) << "dw:\n" << dw->Data().TensorType<2>();
  
  LOG(INFO) << "dx:\n" << dx->DebugString();
  LOG(INFO) << "dx:\n" << dx->Data().TensorType<2>();
  
  LOG(INFO) << "db:\n" << db->DebugString();
}

void exec_simple_calc() {
  const char* pbfile = "./unittest/conf/pb_node_list.graph";
  NodeManagerPtr node_mgr = ParseNodeList(pbfile);

  std::vector<Node*> output_nodes;
  NodePtr y = node_mgr->Get("y");
  output_nodes.push_back(y.get()); 
  NodePtr y1 = node_mgr->Get("y1");
  output_nodes.push_back(y1.get());

  std::vector<Node*> input_nodes;
  input_nodes.push_back(node_mgr->Get("x2").get());
  input_nodes.push_back(node_mgr->Get("x3").get()); 

  LOG(INFO) << "total nodes number of forward: " << node_mgr->TotalNodes().size();

  Gradients grad;
  std::vector<Node*> rt;
  int result = grad.gradients(output_nodes, input_nodes, rt, node_mgr.get());
  auto dx2 = rt[0];
  auto dx3 = rt[1];

  LOG(INFO) << "total nodes number of forward and reverse: " << node_mgr->TotalNodes().size() 
            << ", reverse node size: " << node_mgr->ReversedNodes().size();; 

  NodePtr x2 = node_mgr->Get("x2");
  NodePtr x3 = node_mgr->Get("x3");
  // TODO init source node value 
  x2->SetValue(2);
  x3->SetValue(3);
  x2->Data().TensorType<2>().setConstant(2);
  x3->Data().TensorType<2>().setConstant(3);

  // executor 
  output_nodes.push_back(dx2);
  output_nodes.push_back(dx3);
  Executor* executor = new Executor(output_nodes);
  executor->Run();

  openmi::Timer timer;
  // result 
  auto y_val = node_mgr->Get("y")->Value();
  auto y1_val = node_mgr->Get("y1")->Value();
  LOG(INFO) << "result y: " << y_val;
  //<< ", tensor:\n" << node_mgr->Get("y")->Data().TensorType<2>();
  LOG(INFO) << "result y1: " << y1_val;
  LOG(INFO) << "result x2: " << node_mgr->Get("x2")->Value()  << ", shape: " << node_mgr->Get("x2")->Data().Shape().DebugString();
  //<< ", tensor:\n" << node_mgr->Get("x2")->Data().TensorType<2>();
  LOG(INFO) << "result x3: " << node_mgr->Get("x3")->Value();
  LOG(INFO) << "result dx2: " << node_mgr->Get(dx2->Name())->Value() << ", tensor:\n" << node_mgr->Get(dx2->Name())->Data().TensorType<2>()(100,100);
  LOG(INFO) << "result dx3: " << node_mgr->Get(dx3->Name())->Value() << ", tensor:\n" << node_mgr->Get(dx3->Name())->Data().TensorType<2>()(100,100);
  LOG(INFO) << "result z: " << node_mgr->Get("z")->Value() ;
  //<< ", tensor:\n" << node_mgr->Get("z")->Data().TensorType<2>();
  LOG(INFO) << "result (z*z): " << node_mgr->Get("(z*z)")->Value() ;
  //<< ", tenosr:\n" << node_mgr->Get("(z*z)")->Data().TensorType<2>();
  
  LOG(INFO) << "time: " << timer.Elapsed();
}
