#include "core/framework/gradients.h"
#include "core/framework/node_manager.h"
#include "openmi/pb/node.pb.h"
#include "base/protobuf_op.h"
#include "base/logging.h"
#include <unistd.h>
#include <sys/syscall.h>

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

int main(int argc, char** argv) {
  const char* pbfile = "./unittest/conf/pb_node_list.graph";
  NodeManagerPtr node_mgr = ParseNodeList(pbfile);

  std::vector<Node*> output_nodes;
  NodePtr y = node_mgr->Get("y");
  output_nodes.push_back(y.get()); 

  std::vector<Node*> input_nodes;
  input_nodes.push_back(node_mgr->Get("x2").get());
  input_nodes.push_back(node_mgr->Get("x3").get()); 

  LOG(INFO) << "total nodes number of forward: " << node_mgr->TotalNodes().size();

  Gradients grad;
  std::vector<Node*> rt;
  int result = grad.gradients(output_nodes, input_nodes, rt, node_mgr.get());

  LOG(INFO) << "total nodes number of forward and reverse: " << node_mgr->TotalNodes().size(); 

  std::string grad_topo_nodes;
  for (int i = 0; i < rt.size(); ++i) {
    grad_topo_nodes += " --> " + rt[i]->Name();
  }
  LOG(INFO) << grad_topo_nodes;

  LOG(INFO) << "curr gid: " << (unsigned long) syscall(SYS_gettid);

  return 0;
}
