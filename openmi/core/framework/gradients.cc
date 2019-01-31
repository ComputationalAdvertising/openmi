#include <unordered_map>
#include <iostream>

#include "core/framework/gradients.h"
#include "core/framework/node.h"
#include "core/graph/algorithm.h"

using namespace openmi::graph;

namespace openmi {

int Gradients::gradients(std::vector<Node*>& output_nodes, std::vector<Node*>& input_nodes, std::vector<Node*>& grad_node_list, NodeManager* node_manager) {
  for (Node* node: output_nodes) {
    std::vector<Node*>* out_grads_list = new std::vector<Node*>();
    std::string op_name("OneslikeOp");
    std::string grad_node_name(op_name + "(" + node->Name() + ")");
    NodePtr grad_node = node_manager->Create(grad_node_name, 0, op_name, 2);
    // TODO check new_node
    out_grads_list->push_back(grad_node.get());
    node_to_grads_list_mapper_[node] = out_grads_list;
  }

  LOG(INFO) << "\n===========\ngradients find topo sort\n========";
  if (TopoOrderList(output_nodes, topo_order_list_) != 0) {
    LOG(ERROR) << "get topo order list failed.";
    return -1;
  }
  
  LOG(INFO) << "\n===========\ngenerate grad node list\n=========";
  // reverse topo order
  for (int i = topo_order_list_.size() - 1; i >= 0; i--) {
    Node* node = topo_order_list_[i];
    std::vector<Node*>* output_grads_list = node_to_grads_list_mapper_[node];
    Node* grad_node = ReduceSumNodeList(output_grads_list, node_manager);
    LOG(INFO) << "========> node: " << node->Name() <<  " --> its grad node: " << grad_node->Name();
    node_to_output_grad_mapper_[node] = grad_node;
    for (int i = 0; i < node->Inputs().size(); ++i) {
      Node* ni = node->Inputs()[i];
      std::vector<Node*> grad; 
      grad.push_back(grad_node);
      std::vector<Node*>* grads = node->GetOp()->Gradient(node, grad, node_manager);
      
      std::vector<Node*>* grads_list;
      if (node_to_grads_list_mapper_.find(ni) != node_to_grads_list_mapper_.end()) {
        grads_list = node_to_grads_list_mapper_[ni];
      } else {
        grads_list = node_to_grads_list_mapper_[ni] = new std::vector<Node*>();
      }
      grads_list->push_back((*grads)[i]);
      node_to_grads_list_mapper_[ni] = grads_list;
    }
  }

  // Collect result for gradients requested 
  grad_node_list.clear();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    auto it = node_to_output_grad_mapper_.find(input_nodes[i]);
    CHECK(it != node_to_output_grad_mapper_.end()) << input_nodes[i] << " not in node_to_output_grad_mapper_"; 
    grad_node_list.push_back(it->second);
  }
  return 0;    
}

Node* Gradients::ReduceSumNodeList(std::vector<Node*>* node_list, NodeManager* node_manager) {
  if (node_list == NULL) {
    LOG(ERROR) << "node_list is nullptr";
    return NULL;
  }
  if (node_manager == NULL) {
    LOG(ERROR) << "node manager is null";
    return NULL;
  }

  if ((*node_list).size() == 1) {
    return (*node_list)[0];
  }

  std::string op_name("AddOp");
  Node* prev_node = node_list->at(0);
  Node* rt = NULL;
  for (size_t i = 1; i < node_list->size(); ++i) { 
    Node* curr_node = node_list->at(i);
    std::string node_name("(" + prev_node->Name() + "+" + curr_node->Name() + ")");
    NodePtr grad_node = node_manager->GetOrCreate(node_name, 0, op_name, 2);  // compute_type=2
    if (grad_node == nullptr) {
      LOG(ERROR) << "create reverse node failed. name: " << node_name;
      return NULL;
    }
    grad_node->AddInput(prev_node);
    grad_node->AddInput(curr_node);

    rt = grad_node.get();
    prev_node = grad_node.get();
  }

  return rt;
}
  
} // namespace openmi
