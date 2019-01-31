#ifndef OPENMI_CORE_FRAMEWORK_OP_H_
#define OPENMI_CORE_FRAMEWORK_OP_H_

#include <memory>
#include <string>
#include <vector>
#include "base/logging.h"
#include "core/framework/op_factory.h"

using namespace openmi;

namespace openmi {

class Node;
class NodeManager;

class Op {
public:
  explicit Op(std::string op_name);

  virtual ~Op();

  virtual void Compute(Node* node, std::vector<Node*>& input_nodes);

  virtual std::vector<Node*>* Gradient(Node* node, std::vector<Node*>& output_nodes, NodeManager* node_manager);

  std::string Name() { return op_name_; }

  std::string DebugString() { return op_name_; }

protected:
  std::string op_name_; 

}; // class Op

typedef std::shared_ptr<Op> OpPtr;

/*
class OpKernelContext {
public:
  OpKernelContext() {}
  ~OpKernelContext() {}
private:
  Device* device_;
  std::vector<Node*> inputs_;
  std::vector<Node*> outputs_;
  NodeManager* node_mgr_;
}; // class OpKernelContext
*/

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_OP_H_
