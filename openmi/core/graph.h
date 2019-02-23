#ifndef OPENMI_CORE_FRAMEWORK_GRAPH_H_
#define OPENMI_CORE_FRAMEWORK_GRAPH_H_ 

#include "openmi/idl/proto/node_def.pb.h"
#include "openmi/idl/proto/graph.pb.h"
#include "op_kernel.h"
#include "status.h"

using namespace openmi::proto;

namespace openmi {

class Graph;
class Node;

class NodeProperties {
public:
  NodeProperties(const NodeDef& node_def,
                 const std::vector<DataType> inputs, 
                 const std::vector<DataType> outputs)
    : node_def(node_def), 
      input_types(inputs),
      output_types(outputs) {}

  NodeDef node_def;
  const std::vector<DataType> input_types;
  const std::vector<DataType> output_types;
}; // class NodeProperties

enum NodeClass {
  NC_UNINITIALIZED
  // TODO 
}; 

struct InputTensor {
  Node* node;
  int index;

  InputTensor(Node* n, int idx): node(n), index(idx) {}
  InputTensor() : node(nullptr), index(0) {}
}; // struct InputTensor

struct OutputTensor {
  Node* node;
  int index;

  OutputTensor(Node* n, int idx): node(n), index(idx) {}
  OutputTensor() : node(nullptr), index(0) {}
}; // struct OutputTensor

class Node {
public:
  Node();

  void Initialize(int id, std::shared_ptr<NodeProperties> props);
  
  bool IsInitialized() const { return initialized_; }
  
  void AddInput(Node* n, int idx);
  
  void Clear();

  NodeProperties* properties() { return props_.get(); }

  const NodeDef& def() const { return props_->node_def; }
  
  void set_op(OpKernel* op) { op_.reset(op); }
  OpKernel* op() { return op_.get(); }

  void set_id(int id) { id_ = id; } 
  int id() { return id_; }

  std::vector<InputTensor>& inputs() { return inputs_; }

private:
  friend class Graph;
  Graph* graph_;
  int id_;
  NodeClass class_;
  std::unique_ptr<OpKernel> op_;
  std::shared_ptr<NodeProperties> props_;
  bool initialized_;

  std::vector<InputTensor> inputs_;
  std::vector<OutputTensor> outputs_;
}; // class Node 

struct NodeInfo {
  NodeInfo(const proto::NodeDef& def, int id): node_def(def), id(id) {}
  const proto::NodeDef node_def;
  int id;
};

class Graph {
public:
  Graph(): name_(""), version_(0) {}

  ~Graph() {}

  Node* AddNode(const NodeInfo& ninfo, Status* status);
  Node* AddNode(const NodeDef& node_def, Status* status);

  Node* FindNode(std::string& node_name);
  
  void ReleaseNode(Node* n);

  void AddInput(std::string name, Node* n, int idx);

  void set_name(std::string& name) { name_ = name; }
  std::string name() const { return name_; }
  void set_version(int version) { version_ = version; }
  int version() const { return version_; }

  std::unordered_map<std::string, Node*>& node_mapper() { return node_mapper_; }

private:
  Node* AllocateNode(std::shared_ptr<NodeProperties> props);

private:
  // Graph name 
  std::string name_;
  // Graph version 
  int version_;
  // Map from node ids to allcated nodes. 
  std::vector<Node*> nodes_;
  // Allocated but free nodes and edges.
  std::vector<Node*> free_nodes_;
  // Reversed node list.
  std::vector<Node*> reversed_nodes_;

  std::unordered_map<std::string, Node*> node_mapper_;

  int num_nodes_;

  // TODO 
}; // class Graph 

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_GRAPH_H_ 
