#ifndef OPENMI_CORE_FRAMEWORK_NODE_MANAGER_H_ 
#define OPENMI_CORE_FRAMEWORK_NODE_MANAGER_H_ 

#include <unordered_map>
#include <vector>

#include "core/framework/node.h"
#include "core/framework/op.h"
#include "openmi/pb/node.pb.h"

using namespace openmi;

namespace openmi {

class NodeManager {
public:
  NodeManager() {}
  
  NodeManager(pb::NodeList& node_list);

  NodePtr Get(const std::string& key);

  NodePtr Create(std::string& name, int id, const std::string& op_name);

  NodePtr Create(std::string& name, int id, const std::string& op_name, NodeType type);

  NodePtr Create(std::string& name, int id, const std::string& op_name, NodeType type, NodeComputeType compute_type);
  
  NodePtr Create(std::string& name, int id, const TensorShape& shape, const std::string& op_name);
  
  NodePtr Create(std::string& name, int id, const TensorShape& shape, const std::string& op_name, NodeType type, NodeComputeType compute_type);

  NodePtr GetOrCreate(std::string& name, int id, const std::string& op_name);

  NodePtr GetOrCreate(std::string& name, int id, const std::string& op_name, NodeType type);

  NodePtr GetOrCreate(std::string& name, int id, const std::string& op_name, NodeType type, NodeComputeType compute_type);
  
  NodePtr GetOrCreate(std::string& name, int id, const TensorShape& shape, const std::string& op_name, NodeType type, NodeComputeType compute_type);

  // TODO NodePtr SetNodeAttr(const std::string& key, const pb::NodeAttr& attr);
  NodePtr SetNodeAttr(const std::string& key);

  NodePtr AddInput(const std::string& key, NodePtr node);

  void AddForwardNodes(NodePtr node) {
    forward_nodes_.emplace_back(node);
  }

  void AddReverseNodes(NodePtr node) {
    reverse_nodes_.emplace_back(node);
  }

  inline std::unordered_map<std::string, NodePtr>& TotalNodes() {
    return node_mapper_;
  }

  inline std::vector<NodePtr>& SourceNodes() {
    return source_nodes_;
  }

  inline std::vector<NodePtr>& ForwardNodes() {
    return forward_nodes_;
  }

  inline std::vector<NodePtr>& ReversedNodes() {
    return reverse_nodes_;
  }

/*
public:
  static int DEFAULT_NODE_ID = 0;  // increase
*/
private:
  std::unordered_map<std::string, NodePtr> node_mapper_;
  std::vector<NodePtr> source_nodes_;
  // forward compute node 
  std::vector<NodePtr> forward_nodes_;
  // reverse compute node
  std::vector<NodePtr> reverse_nodes_;
}; // class NodeManager

typedef std::shared_ptr<NodeManager> NodeManagerPtr;

} // namespace openmi  

#endif // OPENMI_CORE_FRAMEWORK_NODE_MANAGER_H_
