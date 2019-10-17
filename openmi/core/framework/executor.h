#ifndef OPENMI_CORE_FRAMEWORK_EXECUTOR_H_
#define OPENMI_CORE_FRAMEWORK_EXECUTOR_H_ 

#include <memory>
#include <unordered_map>
#include "openmi/core/graph/graph.h"
#include "openmi/core/graph/graph_constructor.h"
#include "openmi/idl/proto/graph.pb.h"
#include "status.h"
#include "session_state.h"
#include "base/logging.h"
#include "gradients.h"

extern bool is_training;

namespace openmi {

class Executor {
public:
  explicit Executor(proto::GraphDef& gdef);
  ~Executor();

  void Init(proto::GraphDef& gdef);
  
  void Destroy();

  SessionState* GetSessionState();

  Graph* GetGraph();

  proto::GraphDef& GetGraphDef();

  Status Run();
  
private:
  void InitSessionState();
  void InitComputeOp();

private:
  proto::GraphDef gdef_;
  std::shared_ptr<Graph> g_;
  std::shared_ptr<Gradients> gradients_;
  std::shared_ptr<SessionState> session_state_;
  std::unordered_map<std::string, OpKernelContext*> node_kernel_context_mapper_;
  // todo optimized. support parallel compute when no-dependency node
  std::vector<Node*> compute_nodes_;  
}; // class Executor 

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_EXECUTOR_H_ 
