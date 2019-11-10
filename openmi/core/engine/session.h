#ifndef OPENMI_CORE_ENGINE_SESSION_H_
#define OPENMI_CORE_ENGINE_SESSION_H_ 

#include <unordered_set>
#include "base/bit_op.h" // GetFid, GetNodeIdx, GetRowIdx
#include "openmi/idl/proto/communication.pb.h"
#include "openmi/idl/proto/engine.pb.h"
#include "openmi/idl/proto/graph.pb.h"
#include "openmi/idl/proto/instance.pb.h"
#include "openmi/core/framework/executor.h"
#include "openmi/core/engine/nn_structure.h"
#include "openmi/core/engine/ps_interactor.h"

using namespace openmi;

namespace openmi {

typedef std::shared_ptr<proto::internal::ColumnNode> ColumnNodePtr;
typedef std::shared_ptr<proto::internal::ColumnWeightSchema> ColumnWeightSchemaPtr;
typedef std::shared_ptr<proto::internal::ColumnKeyIndex> ColumnKeyIndexPtr;
typedef std::shared_ptr<proto::comm::ValueList> ValListPtr;
typedef std::shared_ptr<Executor> ExecutorPtr;
typedef std::shared_ptr<proto::Instances> InstancesPtr;
typedef std::shared_ptr<proto::internal::ModelWeightSchema> ModelWeightSchemaPtr;
static int graph_cache_time = 10*60;

class Session {
public: 
  Session();
  ~Session();

  int Init(proto::GraphDef& gdef);
  
  void Destroy();
  
  ExecutorPtr GetExecutor() {
    return executor_;
  }

  int Run(InstancesPtr& batch, bool is_training);   // train or pred
  int Pull();
  int Push();

  // source node 包括column和nn node
  int FeedSourceNode(bool is_training);
  int GetGradient();

private: 
  int ParseGraphSourceNode();
  int CheckColumnNode();
  
  // for sparse feature data 
  int FeedColumnNode(uint32_t colid);
  int FeedNNNode(engine::NNVariableInfo& nn_variable);
  int FeedLabelNode();

  int FeedXNode(ColumnNodePtr& column_node, std::vector<float>& values);
  int FeedWNode(ColumnNodePtr& column_node, ColumnKeyIndexPtr& fids);

  int GetColumnGradient(int colid);
  int GetNNGradient(engine::NNVariableInfo& nn_variable);

  uint64_t NNFid(int node_id, int row_idx);

private:
  ExecutorPtr executor_;
  PsInteractorPtr ps_interactor_;
  InstancesPtr batch_;
  std::unordered_set<uint64_t> fid_set_;
  std::vector<openmi::engine::NNEntry> nn_entrys_;
  std::vector<int> valid_columns_;
  std::unordered_set<int> valid_columns_set_;
  std::unordered_map<int, ColumnNodePtr> column2node_;
  std::unordered_map<std::string, Tensor*> node2tensor_;
  std::unordered_map<int, ColumnKeyIndexPtr> column2keyindex_;
  // column/node id to model weight schema defination.
  std::unordered_map<int, ModelWeightSchemaPtr> model_weight_schema_;

  //std::array<proto::internal::ColumnKeyIndexList*, 1000> column_key_indexes_;
  std::vector<openmi::engine::NNVariableInfo> nn_variables_;
  std::vector<std::string> label_node_;

  bool update_graph_cache_ = true;
  time_t last_update_time_ = 0;
}; // class Session

} // namespace openmi
#endif // OPENMI_CORE_ENGINE_SESSION_H_