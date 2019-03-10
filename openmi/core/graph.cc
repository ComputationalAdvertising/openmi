#include "graph.h"
#include "op_registry.h"
using namespace openmi;

namespace openmi {

Node::Node(): id_(-1), props_(nullptr) {}

void Node::Initialize(int id, std::shared_ptr<NodeProperties> props) {
  id_ = id;
  props_ = std::move(props);
  // attr value(s) of node 
  for (auto& attr: props_->node_def.attr()) {
    LOG(INFO) << "Initialized node attr key:" << attr.first << ", v:" << attr.second.DebugString();
    this->attr_[attr.first].FromProto(attr.second);
  }
  OpKernel* op_kernel;
  OpRegistry::Instance().LookUp(*this, &op_kernel);
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

void Node::AddInput(const std::string in, int idx) {
  input_names_.push_back(in);
}

void Node::AddOutput(const std::string out, int idx) {
  output_names_.push_back(out);
}

void Node::Clear() {
  id_ = -1;
  class_ = NC_UNINITIALIZED;
  props_.reset();
  op_.reset();
  initialized_ = false;
}

Node* Graph::AddNode(const NodeInfo& ninfo, Status* status) {
  Node* n = AddNode(ninfo.node_def, status);
  n->set_id(ninfo.id);
  return n;
}

Node* Graph::AddNode(const proto::NodeDef& node_def, Status* status) {
  LOG(INFO) << "Graph::AddNode begin";
  CHECK(node_mapper_.find(node_def.name()) == node_mapper_.end()) 
    << node_def.name() << " already exists.";
  // TODO OpDef参数与NodeDef参数匹配 
  std::vector<DataType> inputs;
  std::vector<DataType> outputs;
  // status->Update(InOutTypesForNode(node_def, *op_def, &inputs, &outputs));
  if (!status->ok()) {
    return nullptr;
  }

  LOG(INFO) << "Graph::AddNode middle";
  Node* node = AllocateNode(std::make_shared<NodeProperties>(node_def, inputs, outputs));
  LOG(INFO) << "Graph::AddNode middle 1";
  CHECK(node != nullptr) << "add node failed. node:" << node_def.name();
  node_mapper_.insert(std::make_pair(node_def.name(), node));
  LOG(INFO) << "Graph::AddNode done";
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
  LOG(INFO) << "AllocateNode begin";
  Node* node = nullptr;
  if (free_nodes_.empty()) {
    node = new Node;
  } else {
    node = free_nodes_.back();
    free_nodes_.pop_back();
  }
  LOG(INFO) << "AllocateNode middle";
  
  node->graph_ = this;
  const int id = nodes_.size();
  node->Initialize(id, std::move(props));
  nodes_.push_back(node);
  ++num_nodes_;
  LOG(INFO) << "AllocateNode done";
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
  iter->second->AddInput(n->def().name(), idx);
}

}
