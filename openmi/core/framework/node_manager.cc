#include "core/framework/node_manager.h"
#include "core/ops/add_op.h"
#include "core/ops/multiply_op.h"


namespace openmi {

NodeManager::NodeManager(pb::NodeList& nodes_pb) {
  // initialize all node
  for (int i = 0; i < nodes_pb.node().size(); ++i) {
    pb::Node node_def = nodes_pb.node(i);

    Op* op = openmi::Register<OpFactory>::Find(node_def.op())->func();
    if (op == nullptr) {
      LOG(INFO) << "op is null. op: " << node_def.op();
      return;
    }

    NodePtr node = std::make_shared<Node>(node_def.name(), node_def.id());
    node->SetOp(op);
    std::string key = node->Name();
    CHECK(node_mapper_.find(key) == node_mapper_.end()) << key << " already exists!";
    node_mapper_[key] = node;
  } 

  LOG(INFO) << "init all nodes done.";

  // parse compute dependencys. 
  // Note: input nodes of current node must be first occur in pb.file
  for (int i = 0; i < nodes_pb.node().size(); ++i) {
    pb::Node cur_node_def = nodes_pb.node(i);
    std::string key = cur_node_def.name();
    for (int j = 0; j < cur_node_def.inputs().size(); ++j) {
      std::string input_node_key = cur_node_def.inputs(j);
      auto it = node_mapper_.find(input_node_key);
      CHECK(it != node_mapper_.end()) << input_node_key << " not exists!";
      
      AddInput(key, it->second);
    }
  } 
}

NodePtr NodeManager::Get(const std::string& key) {
  auto it = node_mapper_.find(key);
  CHECK(it != node_mapper_.end()) << key << " not exists!";
  return it->second;
}

NodePtr NodeManager::Create(std::string& name, int id, const std::string& op_name) {
  std::string key = name;
  auto it = node_mapper_.find(key);
  CHECK(it == node_mapper_.end()) << key << " already exist!";
  Op* op = openmi::Register<OpFactory>::Find(op_name)->func();
  auto node = node_mapper_[key] = std::make_shared<Node>(name, id);
  node->SetOp(op);
  return node;
}

NodePtr NodeManager::Create(std::string& name, int id, const std::string& op_name, int compute_type) {
  NodePtr node = Create(name, id, op_name);
  node->SetComputeType(compute_type);
  if (compute_type == 1) {
    AddForwardNodes(node);
  } else if (compute_type == 2) {
    AddReverseNodes(node);
  }
  return node;
}

NodePtr NodeManager::GetOrCreate(std::string& name, int id, const std::string& op_name) {
  std::string key = name;
  auto it = node_mapper_.find(key);
  if (it != node_mapper_.end()) {
    return it->second;
  } else {
    Op* op = openmi::Register<OpFactory>::Find(op_name)->func();
    auto node = node_mapper_[key] = std::make_shared<Node>(name, id);
    node->SetOp(op);
    return node;
  }
}

NodePtr NodeManager::GetOrCreate(std::string& name, int id, const std::string& op_name, int compute_type) {
  NodePtr node = GetOrCreate(name, id, op_name);
  if (node->GetComputeType() == -1) {
    node->SetComputeType(compute_type);
    if (compute_type == 1) {
      AddForwardNodes(node);
    } else if (compute_type == 2) {
      AddReverseNodes(node);
    }
  }
  return node;
}

NodePtr NodeManager::SetNodeAttr(const std::string& key) {
  auto node = Get(key);
  // TODO SetAttr
  return node;
}

NodePtr NodeManager::AddInput(const std::string& key, NodePtr node) {
  auto it = node_mapper_.find(key);
  CHECK(it != node_mapper_.end()) << key << " not exists!";
  it->second->AddInput(node.get());
  return it->second;
}
  
}
