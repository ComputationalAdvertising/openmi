#ifndef OPENMI_CORE_FRAMEWORK_EXECUTOR_H_
#define OPENMI_CORE_FRAMEWORK_EXECUTOR_H_ 

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

  SessionState session_state_;
  Graph g_;

private:
  void InitSessionState();
  Gradients gradients_;
}; // class Executor 

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_EXECUTOR_H_ 
