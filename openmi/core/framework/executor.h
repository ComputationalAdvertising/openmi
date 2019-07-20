#ifndef OPENMI_CORE_FRAMEWORK_EXECUTOR_H_
#define OPENMI_CORE_FRAMEWORK_EXECUTOR_H_ 

#include <memory>
#include <unordered_map>
#include "graph.h"
#include "graph_constructor.h"
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

  Status Run();

  SessionState* GetSessionState();

  Graph* GetGraph();
  
private:
  void Init(proto::GraphDef& gdef);
  void InitSessionState();
  void InitComputeOp();
  void Destroy();

private:
  std::shared_ptr<Graph> g_;
  std::shared_ptr<Gradients> gradients_;
  std::shared_ptr<SessionState> session_state_;
  std::unordered_map<std::string, OpKernelContext*> node_kernel_context_mapper_;
  std::vector<Node*> compute_nodes_;
}; // class Executor 

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_EXECUTOR_H_ 
