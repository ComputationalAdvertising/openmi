#include "gradients.h"
#include "algorithm.h"
#include "op_registry.h"
#include "gradient_registry.h"
#include "graph_utils.h"
#include "attr_value_utils.h"

namespace openmi {

int Gradients::gradients(std::vector<Node*>& output_nodes, std::vector<Node*>& input_nodes, std::vector<Node*>& reversed_node_list, Graph* g) {
  std::vector<Node*> used_backward_output_nodes;
  for (Node* n: output_nodes) {
    bool used_backward = true;
    GetAttr<bool>(n->attrs(), "used_backward", &used_backward, ::openmi::AttrValue::kBool);
    if (!used_backward) {
      LOG(WARN) << "sink node [" << n->def().name() 
                << "] not need to back propagation.";
      continue;
    }

    used_backward_output_nodes.push_back(n);
  }

  // Oneslike for sink node that used backward
  for (Node* n: used_backward_output_nodes) {
    std::vector<Node*>* n_grad_list = new std::vector<Node*>();

    auto y_name = n->def().name();
    std::string op("Oneslike");
    std::string grad_node_name(op + "(" + y_name + ")");
    proto::NodeDef grad_node_def;
    grad_node_def.set_name(grad_node_name);
    grad_node_def.set_op(op);
    NodeInfo grad_ninfo(grad_node_def, -1, NC_SOURCE, NS_REVERSE);

    Node* grad_n = g->CreateNode(grad_ninfo, *n);
    CHECK(grad_n != nullptr) << "create grad node failed. name:" << grad_node_name;
    
    n_grad_list->push_back(grad_n);
    node_to_grads_list_mapper_.insert({n, n_grad_list});
  }

  // Topo order list for node that compute gradient
  LOG(INFO) << "\n ------------------- gradients find topo sort --------------------";
  if (TopoOrderList(used_backward_output_nodes, topo_order_list_, g) != 0) {
    LOG(ERROR) << "get topo order list failed.";
    return -1;
  }

  LOG(INFO) << "\n ------------------- generate grad node list --------------------";
  LOG(INFO) << "topo_order_list_.size: " << topo_order_list_.size();

  // Create gradient node from sink to source by GradConstructor
  for (int i = topo_order_list_.size() - 1; i >= 0; i--) {
    Node* n = topo_order_list_.at(i);
    DLOG(INFO) << "n.name: " << n->def().name();
    std::vector<Node*>* n_grad_list = node_to_grads_list_mapper_[n];
    CHECK(n_grad_list != nullptr) << "grad list is null. node:" << n->def().name();
    Node* grad_node = SumGradients(*n_grad_list, g);
    CHECK(grad_node != nullptr) << " gradient node is null. n.name: " << n->def().name();
    DLOG(INFO) << "------> node:" << n->def().name() << ", it grad node: " << grad_node->def().name();
    node_to_output_grad_mapper_.insert({n, grad_node});
    
    std::vector<Node*> dy;
    dy.push_back(grad_node);
    std::vector<Node*> dx_list;
 
    GradConstructor grad_constructor;
    GradientRegistry::Instance().Lookup(n->def().op(), &grad_constructor);
    grad_constructor(*n, dy, dx_list, *g);

    DLOG(INFO) << "its input size: " << n->inputs().size();
    if (n->outputs().size() > 0) {
      DLOG(INFO) << "n.outputs[0]: " << n->outputs().at(0);
    }
    for (int j = 0; j < n->inputs().size(); ++j) {
      Node* nj = g->FindNode(n->inputs().at(j));
      std::vector<Node*>* nj_grad_list;
      auto it = node_to_grads_list_mapper_.find(nj);
      if (it == node_to_grads_list_mapper_.end()) {
        nj_grad_list = new std::vector<Node*>();
        node_to_grads_list_mapper_.insert({nj, nj_grad_list});
      } else {
        nj_grad_list = it->second;
      }
      nj_grad_list->push_back(dx_list.at(j));
      node_to_grads_list_mapper_.insert({nj, nj_grad_list});
    }
  }

  LOG(INFO) << "\n ------------------- generate grad node list done --------------------";

  // Collect result for gradient requested 
  reversed_node_list.clear();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    auto it = node_to_output_grad_mapper_.find(input_nodes[i]);
    CHECK(it != node_to_output_grad_mapper_.end())
      << input_nodes[i]->def().name() << " not in node_to_output_grad_mapper_";
    reversed_node_list.push_back(it->second);
  }

  LOG(INFO) << "\n ------------------ gradients done ------------------";

  return 0;
}

// multiply gradient nodes accumulate sum op
Node* Gradients::SumGradients(std::vector<Node*>& node_list, Graph* g) {
  if (node_list.empty()) {
    LOG(ERROR) << "node_list is empty.";
    return nullptr;
  }
  if (g == nullptr) {
    LOG(ERROR) << "graph is null.";
    return nullptr;
  }

  if (node_list.size() == 1) {
    return node_list[0];
  }

  proto::NodeDef grad_node_def;
  std::string op("accumulate_n");
  grad_node_def.set_op(op);
  
  auto in0_name = node_list.at(0)->def().name();
  grad_node_def.add_input(in0_name);

  std::string node_name(op + "_gradient(" + in0_name);
  for (auto i = 1; i < node_list.size(); ++i) {
    auto cur_node_name = node_list.at(i)->def().name();
    node_name += "+" + cur_node_name;
    grad_node_def.add_input(cur_node_name);
  }
  node_name += ")";

  DLOG(DEBUG) << "accumulate_n node name: " << node_name;
  
  grad_node_def.set_name(node_name);

  Node* grad_node = CreateGradNode(grad_node_def, *g, in0_name); 
  return grad_node;
}

} // namespace openmi
