#ifndef OPENMI_CORE_ENGINE_MODEL_PARSER_H_
#define OPENMI_CORE_ENGINE_MODEL_PARSER_H_

#include "openmi/core/framework/executor.h"
#include "openmi/idl/proto/engine.pb.h"
using namespace openmi;

namespace openmi {

class ModelParser {
public: 
  static void CreateModelWeightSchema(Executor* exec, std::unordered_map<int, std::shared_ptr<proto::internal::ModelWeightSchema> >& schema) {
    for (Node* node: exec->GetGraph()->source_nodes()) {
      std::string node_name = node->def().name();
      int column_id = -1;
      GetAttr(node->attrs(), "col_id", &column_id);
      if (column_id == -1) {
        bool label = false;
        GetAttr(node->attrs(), "label", &label);
        // nn source node. 暂不支持不同NN weight对应不同optimizer
        if (!label && node->node_info().node_scope == NS_FORWARD) {
          int node_id = 0;
          GetAttr(node->attrs(), "node_id", &node_id);
          CHECK(node_id < 0) 
            << ", but node_id:" << node_id << ", node name:" << node_name;

          CHECK(schema.find(node_id) == schema.end())
            << "node id has already exists. node_id:" << node_id 
            << ", node_name:" << node_name;

          auto model_weight_schema = std::make_shared<proto::internal::ModelWeightSchema>();
          schema.insert({node_id, model_weight_schema});
          InitModelWeightSchema(exec, node_name, node_id, model_weight_schema.get());
        }
      } else {
        SourceNodeType source_node_type = proto::X;
        GetAttr(node->attrs(), "source_node_type", &source_node_type);
        if (source_node_type != proto::W 
          || node->node_info().node_scope != NS_FORWARD) {
          continue;
        }
        std::shared_ptr<proto::internal::ModelWeightSchema> model_weight_schema;
        if (schema.find(column_id) != schema.end()) {
          model_weight_schema = schema[column_id];
        } else {
          model_weight_schema = std::make_shared<proto::internal::ModelWeightSchema>();
          schema.insert({column_id, model_weight_schema});
        }
        int embedding_size = 1;
        GetAttr(node->attrs(), "embedding_size", &embedding_size);
        CHECK(embedding_size > 0) << "node '" << node_name 
          << "' has invalid param 'embedding_size' that value should be greater 0";
        InitModelWeightSchema(node_name, column_id, model_weight_schema.get(), embedding_size);
      }
    }
  }

  static void InitModelWeightSchema(std::string& node_name, int id, proto::internal::ModelWeightSchema* schema, int size) {
    schema->set_column_id(id);
    auto* weight_schema = schema->add_weight_schema();
    auto cur_weight_len = schema->weight_len();
    weight_schema->set_offset(cur_weight_len);
    weight_schema->set_size(size);
    schema->set_weight_len(cur_weight_len + size);
    LOG(INFO) << __FUNCTION__ << " node_name:" << node_name << ", id:" << id << ", size:" << size 
      << ", weight_len:" << schema->weight_len() << ", number of weight:" << schema->weight_schema_size();
  }

  static void InitModelWeightSchema(Executor* exec, std::string& node_name, int id, proto::internal::ModelWeightSchema* schema) {
    schema->set_column_id(id);
    auto* weight_schema = schema->add_weight_schema();

    Tensor* t = nullptr;
    exec->GetSessionState()->GetTensor(node_name, &t);
    CHECK(t != nullptr) << "tensor not found from session state. node name: " << node_name;
    CHECK(t->shape().Dims() == 2);
    auto dim_2nd = t->shape().DimSize(1);
    auto cur_weight_len = schema->weight_len();
    weight_schema->set_offset(cur_weight_len);
    weight_schema->set_size(dim_2nd);
    schema->set_weight_len(cur_weight_len + dim_2nd);
    LOG(INFO) << __FUNCTION__ << " node_name:" << node_name << ", id:" << id 
      << ", size:" << dim_2nd << ", weight_len:" << schema->weight_len();
  }
}; // class ModelParser

} 
#endif // OPENMI_CORE_ENGINE_MODEL_PARSER_H_ 