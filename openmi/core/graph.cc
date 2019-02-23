#include "graph.h"
#include "op_registry.h"
using namespace openmi;

namespace openmi {

Node::Node(): id_(-1), props_(nullptr) {}

void Node::Initialize(int id, std::shared_ptr<NodeProperties> props) {
  id_ = id;
  props_ = std::move(props);
  OpKernel* op_kernel;
  OpRegistry::Instance().LookUp(props_->node_def, &op_kernel);
  if (op_kernel == nullptr) {
    LOG(ERROR) << "fetch op from OpRegistry failed. op_name:" << props_->node_def.op();
    return;
  }
  op_.reset(op_kernel);
  initialized_ = true;
}

void Node::AddInput(Node* n, int idx) {
  InputTensor input(n, idx);
  inputs_.push_back(input);
}

void Node::Clear() {
  id_ = -1;
  class_ = NC_UNINITIALIZED;
  props_.reset();
  op_.reset();
  initialized_ = false;
}

/*
Graph::Graph(proto::GraphDef& gdef) {
  name_ = gdef.name();
  version_ = gdef.version();
  Status status;
  for (auto& node_def: gdef.node()) {
    CHECK(node_mapper_.find(node_def.name()) == node_mapper_.end()) 
      << node_def.name() << " already exists.";
    Node* node = AddNode(node_def, &status);
    CHECK(node != nullptr) << "add node failed. node:" << node_def.name();
    node_mapper_.insert({node_def.name(), node});
  }

  // parse dependencies 
  for (auto& node_def: gdef.node()) {
    for (int i = 0; i < node_def.input().size(); ++i) {
      auto ith_input = node_def.input(i);
      auto iter = node_mapper_.find(ith_input);
      CHECK(iter != node_mapper_.end()) << ith_input << " not exists in graphdef.proto.";
      AddInput(node_def.name(), iter->second, i);
    }
  }
}
*/

Node* Graph::AddNode(const NodeInfo& ninfo, Status* status) {
  Node* n = AddNode(ninfo.node_def, status);
  n->set_id(ninfo.id);
  return n;
}

Node* Graph::AddNode(const proto::NodeDef& node_def, Status* status) {
  CHECK(node_mapper_.find(node_def.name()) == node_mapper_.end()) 
    << node_def.name() << " already exists.";
  // TODO OpDef参数与NodeDef参数匹配 
  std::vector<DataType> inputs;
  std::vector<DataType> outputs;
  // status->Update(InOutTypesForNode(node_def, *op_def, &inputs, &outputs));
  if (!status->ok()) {
    return nullptr;
  }

  Node* node = AllocateNode(std::make_shared<NodeProperties>(node_def, inputs, outputs));
  CHECK(node != nullptr) << "add node failed. node:" << node_def.name();
  node_mapper_.insert(std::make_pair(node_def.name(), node));
  return node;
}

Node* Graph::FindNode(std::string& node_name) {
  auto iter = node_mapper_.find(node_name);
  if (iter == node_mapper_.end()) {
    return nullptr;
  }
  return iter->second;
}

Node* Graph::AllocateNode(std::shared_ptr<NodeProperties> props) {
  Node* node = nullptr;
  if (free_nodes_.empty()) {
    node = new Node;
  } else {
    node = free_nodes_.back();
    free_nodes_.pop_back();
  }
  
  node->graph_ = this;
  const int id = nodes_.size();
  node->Initialize(id, std::move(props));
  nodes_.push_back(node);
  ++num_nodes_;
  return node;
}

void Graph::ReleaseNode(Node* node) {
  nodes_[node->id()] = nullptr;
  free_nodes_.push_back(node);
  --num_nodes_;
  node->Clear();
}

void Graph::AddInput(std::string name, Node* n, int idx) {
  auto iter = node_mapper_.find(name);
  CHECK(iter != node_mapper_.end()) << name << " not exists.";
  iter->second->AddInput(n, idx);
}

}
