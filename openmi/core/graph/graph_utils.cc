#include "graph_utils.h"
#include <set>

namespace openmi {

// Create node without related node. for example. SliceOp as GradNode which not respond to forward node
Node* CreateGradNode(proto::NodeDef& ndef, Graph& g, NodeClass nc, NodeScope ns) {
  NodeInfo ninfo(ndef, -1, nc, ns);
  Status s;
  Node* new_node = g.AddNode(ninfo, &s);
  if (!s.ok()) {
    return nullptr;
  }
  return new_node;
}

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
  DLOG(INFO) << "grad node name:" << node_name << ", op:" << op;
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

void FindSinkNodes(Graph* g, std::vector<Node*>& sink_nodes, bool used_back_props) {
  // find output nodes 
  std::set<std::string> exist_outputs;
  std::set<std::string> all_node_keys;
  auto it = g->node_mapper().begin();
  while (it != g->node_mapper().end()) {
    Node* n = it->second;
    all_node_keys.insert(n->def().name());
    for (size_t idx = 0; idx < n->inputs().size(); ++idx) {
      exist_outputs.insert(n->inputs()[idx]);
    }
    it++;
  }

  std::set<std::string>::iterator iter;
  for (iter = all_node_keys.begin(); iter != all_node_keys.end(); iter++) {
    if (exist_outputs.find(*iter) == exist_outputs.end()) {
      auto it = g->node_mapper().find(*iter);
      CHECK(it != g->node_mapper().end()) << *iter << " not in node_mapper in graph";
      sink_nodes.push_back(it->second);
    }
  }
}

void DebugGraphNodes(Graph* g) {
  DLOG(INFO) << "global sink nodes info.";
  for (int i = 0; i < g->global_sink_nodes().size(); ++i) {
    DLOG(INFO) << "global sink nodes i[" << i 
               << "] node_name: " << g->global_sink_nodes().at(i)->def().name();
    if (g->global_sink_nodes().at(i)->outputs().size() > 0) {
      DLOG(INFO) << "outputs(0): " << g->global_sink_nodes().at(i)->outputs().at(0);
    }
  }
    
  DLOG(INFO) << "after topology order list.";
  for (int i = 0; i < g->global_topo_nodes().size(); ++i) {
    DLOG(INFO) << "global topo node. i[" << i 
               << "], node_name: " << g->global_topo_nodes().at(i)->def().name();
  }
}

}
