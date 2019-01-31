#include "core/framework/node_manager.h"
#include "core/framework/gradients.h"
#include "openmi/pb/node.pb.h"
#include "base/protobuf_op.h"

using namespace openmi;

void PrintNode(Node* n) {
  LOG(INFO) << n->DebugString();
  if (n->Inputs().size() == 0) return;
  for (Node* node: n->Inputs()) {
    PrintNode(node);
  }
}

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
  /////// protobuf parse 
  LOG(INFO) << "\nparse protobuf node_list\n";
  const char* pbfile = "./unittest/conf/pb_node_list.graph";
  NodeManagerPtr node_mgr2 = ParseNodeList(pbfile);
  LOG(INFO) << "\n-------------- y -------------\n";
  std::string y_key("y");
  NodePtr y = node_mgr2->Get(y_key);
  PrintNode(y.get());
  LOG(INFO) << "\n-------------- y1 -------------\n";
  std::string y1_key("y1");
  NodePtr y1 = node_mgr2->Get(y1_key);
  PrintNode(y1.get()); 

  return 0;
}
