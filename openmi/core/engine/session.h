#ifndef OPENMI_CORE_ENGINE_SESSION_H_
#define OPENMI_CORE_ENGINE_SESSION_H_ 

#include "openmi/idl/proto/engine.pb.h"
#include "openmi/idl/proto/graph.pb.h"
#include "openmi/core/framework/executor.h"
using namespace openmi;

namespace openmi {

typedef std::shared_ptr<proto::internal::ColumnNode> ColumnNodePtr;
typedef std::shared_ptr<Executor> ExecutorPtr;

class Session {
public: 
  Session();
  ~Session();

  int Init(proto::GraphDef& gdef);
  void Destroy();
  
  ExecutorPtr GetExecutor() {
    return executor_;
  }

private: 
  int GetGraphNode();

private:
  ExecutorPtr executor_;
  std::unordered_map<int, ColumnNodePtr> column2node_;
  std::unordered_map<std::string, Tensor*> node2tensor_;
  std::vector<std::string> forward_nn_variable_;
  std::vector<std::string> reverse_nn_variable_;
  std::vector<std::string> label_node_;
}; // class Session

} // namespace openmi
#endif // OPENMI_CORE_ENGINE_SESSION_H_