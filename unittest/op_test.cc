#include "node_manager.h"
#include "op.h"
#include "base/logging.h"
#include "add_op.h"
#include "multiply_op.h"

using namespace openmi;

int main(int argc, char** argv) {
  NodeManager* node_manager = new NodeManager();
  std::string node1_name("n1");
  std::make_shared<AddOp>(node_manager, node1_name);

  std::string node2_name("n2");
  std::make_shared<AddOp>(node_manager, node2_name); 

  auto node_mapper = node_manager->TotalNodes();
  LOG(INFO) << "node_mapper.size: " << node_mapper.size();
  auto it = node_mapper.begin();
  while (it != node_mapper.end()) {
    LOG(INFO) << it->first << ", " << it->second->DebugString();
    std::string new_node(it->first + "_grad");
    std::vector<Node*> output_nodes;
    LOG(INFO) << "output_nodes.size: " << output_nodes.size();
    LOG(INFO) << "op_name: " << it->second.get()->GetOp()->Name();
    it->second->GetOp()->Gradient(it->second.get(), output_nodes, node_manager);
    it++;
  }
  
  node_mapper = node_manager->TotalNodes();
  LOG(INFO) << "node_mapper.size: " << node_mapper.size();
  it = node_mapper.begin();
  while (it != node_mapper.end()) {
    LOG(INFO) << it->first << ", " << it->second->DebugString();
    std::string new_node(it->first + "_grad");
    //it->second->GetOp()->Gradient(new_node, node_manager);
    it++;
  }
}
