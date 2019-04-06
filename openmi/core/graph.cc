#include "graph.h"
#include "op_registry.h"
using namespace openmi;

namespace openmi {

void Node::Initialize(NodeInfo& ninfo) {
  LOG(INFO) << "Node::Initialize node:" << ninfo.node_def.name();
  ninfo_ = ninfo;
  // attr value(s) of node 
  for (auto& attr: ninfo_.node_def.attr()) {
    this->attr_[attr.first].FromProto(attr.second);
  }
  OpKernel* op_kernel;
  OpRegistry::Instance().LookUp(*this, &op_kernel);
  CHECK(op_kernel != nullptr) 
    << "lookup op from registry failed. op_name:" << ninfo_.node_def.op();
  op_.reset(op_kernel); 

  initialized_ = true;
}

void Node::AddInput(const std::string in) {
  input_names_.push_back(in);
}

void Node::AddOutput(const std::string out) {
  output_names_.push_back(out);
}

void Node::Clear() {
  op_.reset();
  ninfo_.id = -1;
  ninfo_.node_class = NC_UNINITIALIZED;
  initialized_ = false;
}

Node* Graph::AddNode(NodeInfo& ninfo, Status* status) {
  auto node_name = ninfo.node_def.name();
  LOG(INFO) << "AddNode name:" << node_name;
  CHECK(node_mapper_.find(node_name) == node_mapper_.end()) 
    << ninfo.node_def.name() << " already exists.";
  
  Node* node = AllocateNode(ninfo);
  CHECK(node != nullptr) << "add node failed. node:" << node_name;
  node_mapper_.insert({node_name, node});

  if (ninfo.node_scope == NS_REVERSE) {
    reversed_nodes_.push_back(node);
  }

  nodes_.push_back(node);
  ++num_nodes_;

  // model parameter nodes
  if (ninfo.node_def.op() == "Variable") {
    variable_nodes_.push_back(node);
  }

  LOG(INFO) << "AddNode name:" << node_name;
  return node;
}

Node* Graph::GetOrCreateNode(NodeInfo& new_ninfo, Node& related_node) {
  auto node_name = new_ninfo.node_def.name();
  auto it = node_mapper_.find(node_name);
  if (it != node_mapper_.end()) {
    return it->second;
  }
  return CreateNode(new_ninfo, related_node);
}

Node* Graph::CreateNode(NodeInfo& new_ninfo, Node& related_node) {
  new_ninfo.node_def.set_device(related_node.def().device());
  new_ninfo.related_node_name = related_node.def().name();
  related_node.node_info().related_node_name = new_ninfo.node_def.name();
  LOG(INFO) << "new_info: " << new_ninfo.node_def.name() << ", related: " << related_node.def().name() << ", n: " << related_node.node_info().related_node_name;
  Status s;
  Node* new_node = AddNode(new_ninfo, &s);
  if (!s.ok()) {
    return nullptr;
  }
  return new_node;
}

Node* Graph::FindNode(const std::string& node_name) {
  auto iter = node_mapper_.find(node_name);
  if (iter == node_mapper_.end()) {
    return nullptr;
  }
  return iter->second;
}

Node* Graph::AllocateNode(NodeInfo& ninfo) {
  Node* node = nullptr;
  if (free_nodes_.empty()) {
    node = new Node;
  } else {
    node = free_nodes_.back();
    free_nodes_.pop_back();
  }

  node->graph_ = this;
  ninfo.id = nodes_.size();
  node->Initialize(ninfo);
  return node;
}

void Graph::ReleaseNode(Node* node) {
  nodes_[node->id()] = nullptr;
  free_nodes_.push_back(node);
  --num_nodes_;
  node->Clear();
}

void Graph::AddInput(std::string name, Node* n) {
  auto iter = node_mapper_.find(name);
  CHECK(iter != node_mapper_.end()) << name << " not exists.";
  iter->second->AddInput(n->def().name());
}

}
