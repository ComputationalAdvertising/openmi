#ifndef OPENMI_CORE_ENGINE_SESSION_H_
#define OPENMI_CORE_ENGINE_SESSION_H_ 

#include "base/bit_op.h" // GetFid, GetNodeIdx, GetRowIdx
#include "openmi/idl/proto/instance.pb.h"
#include "openmi/idl/proto/engine.pb.h"
#include "openmi/idl/proto/graph.pb.h"
#include "openmi/core/framework/executor.h"
using namespace openmi;

namespace openmi {

typedef std::shared_ptr<proto::internal::ColumnNode> ColumnNodePtr;
typedef std::shared_ptr<proto::internal::ColumnWeightSchema> ColumnWeightSchemaPtr;
typedef std::shared_ptr<Executor> ExecutorPtr;
typedef std::shared_ptr<proto::Instances> InstancesPtr;

class Session {
public: 
  Session();
  ~Session();

  int Init(proto::GraphDef& gdef);
  void Destroy();
  
  ExecutorPtr GetExecutor() {
    return executor_;
  }

  int Run();

  // source node 包括column和nn node
  int FeedSourceNode(InstancesPtr& batch, std::unordered_map<uint64_t, proto::internal::ValList>& model_weights);

  int FeedColumnNode(InstancesPtr& batch, uint32_t colid);
  // column node
  int FeedColumnNode(InstancesPtr& batch, uint32_t colid, std::unordered_map<uint64_t, proto::internal::ValList>& model_weights);
  // nn node
  int FeedNNNode(int node_idx, std::string& nn_node_name, std::unordered_map<uint64_t, proto::internal::ValList>& model_weights);

  int GetGradientResult();

private: 
  int ParseGraphSourceNode();
  int CheckColumnNode();

private:
  ExecutorPtr executor_;
  std::vector<int> valid_columns_;
  std::unordered_map<int, ColumnNodePtr> column2node_;
  std::unordered_map<std::string, Tensor*> node2tensor_;
  std::vector<std::string> forward_nn_variable_;
  std::vector<std::string> reverse_nn_variable_;
  std::vector<std::string> label_node_;
}; // class Session

} // namespace openmi
#endif // OPENMI_CORE_ENGINE_SESSION_H_