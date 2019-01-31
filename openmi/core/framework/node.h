#ifndef OPENMI_CORE_FRAMEWORK_NODE_H_
#define OPENMI_CORE_FRAMEWORK_NODE_H_ 

#include <string>
#include <vector>
#include "core/framework/op.h"
#include "core/framework/tensor.h"
#include "base/logging.h"
using namespace openmi;

namespace openmi {

class Node {
public:
  //Node(pb::NodeAttr& attr)
  Node(std::string name, int id): name_(name), id_(id), compute_type_(-1) {
    value_ = 0;
    std::string shape("1000,2000");
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

  void SetComputeType(int type) {
    compute_type_ = type;
  }

  int GetComputeType() {
    return compute_type_;
  }

  std::string DebugString() const {
    std::string ret = "name:" + name_ + ", its input: ";
    for (int i = 0; i < inputs_.size(); ++i) {
      ret += inputs_[i]->Name() + ",";
    }
    return ret;
  }

private:
  std::string name_;
  int id_;
  Op* op_ = nullptr;
  openmi::Tensor<float>* tensor_;
  std::vector<Node*> inputs_;
  // TODO for test 
  int value_;
  int compute_type_;    // 0: placeholder,1:forward,2:reverse
}; // class Node

typedef std::shared_ptr<Node> NodePtr;

} // namespace openmi 

#endif // OPENMI_CORE_FRAMEWORK_NODE_H_
