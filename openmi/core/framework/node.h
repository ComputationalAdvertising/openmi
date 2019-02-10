#ifndef OPENMI_CORE_FRAMEWORK_NODE_H_
#define OPENMI_CORE_FRAMEWORK_NODE_H_ 

#include <string>
#include <vector>
#include "base/logging.h"
#include "core/framework/op.h"
#include "core/framework/tensor.h"
#include "openmi/pb/node.pb.h"
using namespace openmi;

namespace openmi {

enum NodeType {
  NT_UNINITIALIZED,
  NT_SOURCE,
  NT_SINK,
  NT_OP,          // except for PlaceholderOp
  NT_EXIT
};

enum NodeComputeType {
  NCT_NOTHING,
  NCT_FORWARD,
  NCT_REVERSE
};

class Node {
public:
  //Node(pb::NodeAttr& attr) 
  Node(pb::Node& node_def);

  Node(std::string name, int id, const TensorShape& shape); 
  
  Node(std::string name, int id) 
    : name_(name), 
      id_(id), 
      type_(NT_UNINITIALIZED), 
      compute_type_(NCT_NOTHING) {
    value_ = 0;
    std::string shape("3,10");
    tensor_ = new openmi::Tensor<float>(shape);
  }

  ~Node() {
    if (op_ != nullptr) {
      delete op_; op_ = nullptr;
    }

    /*
    if (tensor_ != nullptr) {
      delete tensor_; tensor_ = nullptr;
    }
    */
  }

  void SetName(std::string& name) { name_ = name; }

  std::string Name() { return name_; } 

  void SetId(int id) { id_ = id; } 

  int Id() { return id_; }

  void SetOp(Op* op) { 
    // TODO CHECK_EQ()
    if (op == nullptr) {
      LOG(ERROR) << "op is null";
      return;
    }
    op_ = op;
    if (op->Name()=="PlaceholderOp" || op->Name() == "ZeroslikeOp") {
      type_ = NT_SOURCE;
    } else {
      type_ = NT_OP;
    }
  }

  Op* GetOp() { return op_; }

  void AddInput(Node* node) {
    inputs_.push_back(node);
  }

  std::vector<Node*> Inputs() { 
    return inputs_; 
  }

  void SetValue(int v) {
    value_ = v;
  }

  int& Value() { return value_; }

  openmi::Tensor<float>& Data() { 
    return *tensor_; 
  }

  void SetType(NodeType ntype) {
    type_ = ntype;
  }

  NodeType GetType() {
    return type_;
  }

  void SetComputeType(NodeComputeType nctype) {
    compute_type_ = nctype;
  }

  NodeComputeType GetComputeType() {
    return compute_type_;
  }

  inline bool IsSource() { return type_ == NT_SOURCE; }
  inline bool IsSink() { return type_ == NT_SINK; }
  inline bool IsOp() { return type_ == NT_EXIT; }
  inline bool IsExit() { return type_ == NT_EXIT; }
  inline bool IsForwardNode() { return compute_type_ == NCT_FORWARD; }
  inline bool IsReverseNode() { return compute_type_ == NCT_REVERSE; }

  std::string DebugString() const {
    std::string ret = "name:" + name_ + ", shape:" + tensor_->Shape().DebugString() + ", its input: ";
    for (int i = 0; i < inputs_.size(); ++i) {
      ret += inputs_[i]->Name() + ",";
    }
    return ret;
  }

private:
  std::string name_;
  int id_;
  NodeType type_;
  NodeComputeType compute_type_;
  Op* op_ = nullptr;
  openmi::Tensor<float>* tensor_;
  std::vector<Node*> inputs_;
  // TODO for test 
  int value_;
}; // class Node

typedef std::shared_ptr<Node> NodePtr;

} // namespace openmi 

#endif // OPENMI_CORE_FRAMEWORK_NODE_H_
