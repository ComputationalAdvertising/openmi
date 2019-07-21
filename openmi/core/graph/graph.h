#ifndef OPENMI_CORE_FRAMEWORK_GRAPH_H_
#define OPENMI_CORE_FRAMEWORK_GRAPH_H_ 

#include "openmi/idl/proto/node_def.pb.h"
#include "openmi/idl/proto/graph.pb.h"

#include "attr_value.h"
#include "op_kernel.h"
#include "tensor.h"
#include "tensor_shape.h"
#include "status.h"

using namespace openmi::proto;

namespace openmi {

class Graph;
class Node;

enum NodeClass {
  NC_UNINITIALIZED,
  NC_SOURCE,
  NC_OP,
  NC_EXIT
  // TODO 
}; 

enum NodeScope {
  NS_NOTHING,
  NS_FORWARD,
  NS_REVERSE
};

struct NodeInfo {
  NodeInfo() {}

  NodeInfo(proto::NodeDef& def, 
           int id = -1, 
           NodeClass node_class = NC_UNINITIALIZED, 
           NodeScope node_scope = NS_NOTHING,
           std::string related_node_name = "") 
    : node_def(def), 
      id(id),
      node_class(node_class),
      node_scope(node_scope), 
      related_node_name(related_node_name) {}

  proto::NodeDef node_def;
  int id;
  NodeClass node_class;
  NodeScope node_scope;
  std::string related_node_name;
};

class Node {
public:
  Node() {}

  void Initialize(NodeInfo& ninfo);
  
  bool IsInitialized() const { return initialized_; }
  
  void AddInput(const std::string in);
  void AddOutput(const std::string out);
  
  void Clear();

  NodeDef& def() { return ninfo_.node_def; }
  NodeInfo& node_info() { return ninfo_; }

  std::string& related_node_name() {
    return ninfo_.related_node_name;
  }
  
  void set_op(OpKernel* op) { op_.reset(op); }
  OpKernel* op() { return op_.get(); }

  void set_id(int id) { ninfo_.id = id; } 
  int id() { return ninfo_.id; }

  std::unordered_map<std::string, AttrValue>& attrs() {
    return attr_;
  }
  
  std::vector<std::string>& inputs() { return input_names_; }
  std::vector<std::string>& outputs() { return output_names_; }

  std::string DebugString() { return def().DebugString(); }

private:
  friend class Graph;
  Graph* graph_;
  NodeInfo ninfo_;
  std::unique_ptr<OpKernel> op_;
  std::unordered_map<std::string, AttrValue> attr_;
  bool initialized_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
}; // class Node 

class Graph {
public:
  Graph(): name_(""), version_(0) {}

  ~Graph() {}

  void Destroy() {
    // TODO
  }

  Node* AddNode(NodeInfo& ninfo, Status* status);

  // TODO opti api 
  Node* CreateNode(NodeInfo& new_ninfo, Node& related_node);

  Node* GetOrCreateNode(NodeInfo& new_ninfo, Node& related_node);
  
  Node* FindNode(const std::string& node_name);
  
  void ReleaseNode(Node* n);

  void AddInput(std::string name, Node* n);

  void set_name(std::string& name) { name_ = name; }
  std::string name() const { return name_; }

  void set_version(int version) { version_ = version; }
  int version() const { return version_; }

  std::vector<Node*>& nodes() { return nodes_; }
  std::vector<Node*>& source_nodes() { return source_nodes_; }
  std::vector<Node*>& forward_topo_nodes() { return forward_topo_nodes_; }
  std::vector<Node*>& global_topo_nodes() { return global_topo_nodes_; }
  std::vector<Node*>& reversed_nodes() { return reversed_nodes_; }
  std::vector<Node*>& variable_nodes() { return variable_nodes_; }
  std::vector<Node*>& reversed_variable_nodes() { return reversed_variable_nodes_; }
  std::vector<Node*>& sink_nodes() { return sink_nodes_; }
  std::vector<Node*>& global_sink_nodes() { return global_sink_nodes_; }

  std::unordered_map<std::string, Node*>& node_mapper() { return node_mapper_; }


private:
  Node* AllocateNode(NodeInfo& ninfo);

private:
  // Graph name 
  std::string name_;
  // Graph version 
  int version_;
  // All allcated nodes.
  std::vector<Node*> nodes_;
  // Allocated but free nodes and edges.
  std::vector<Node*> free_nodes_;
  // Source nodes that no deps
  std::vector<Node*> source_nodes_;
  // Forward topo order list.
  std::vector<Node*> forward_topo_nodes_;
  // Reversed nodes 
  std::vector<Node*> reversed_nodes_;
  // All topo order nodes. it contains Forward and Reversed topo nodes.
  std::vector<Node*> global_topo_nodes_;
  // Varibale nodes that respond to model parameter to be updated from ps.
  std::vector<Node*> variable_nodes_; 
  // Reversed Variable nodes that respond to gradients of model parameter
  std::vector<Node*> reversed_variable_nodes_; 
  // Sink nodes that not in all node deps
  std::vector<Node*> sink_nodes_;
  std::vector<Node*> global_sink_nodes_;

  std::unordered_map<std::string, Node*> node_mapper_;

  int num_nodes_;

  // TODO 
}; // class Graph 

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_GRAPH_H_ 
