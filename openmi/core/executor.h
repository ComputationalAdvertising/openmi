#ifndef OPENMI_CORE_FRAMEWORK_EXECUTOR_H_
#define OPENMI_CORE_FRAMEWORK_EXECUTOR_H_ 

#include "graph.h"
#include "graph_constructor.h"
#include "openmi/idl/proto/graph.pb.h"
#include "status.h"
#include "session_state.h"
#include "base/logging.h"

namespace openmi {

class Executor {
public:
  explicit Executor(proto::GraphDef& gdef);
  
  ~Executor();

  Status Run();

//private:
  SessionState session_state_;
  Graph g_;
}; // class Executor 

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_EXECUTOR_H_ 
