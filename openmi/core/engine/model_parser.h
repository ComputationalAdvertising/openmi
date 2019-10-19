#ifndef OPENMI_CORE_ENGINE_MODEL_PARSER_H_
#define OPENMI_CORE_ENGINE_MODEL_PARSER_H_

#include "openmi/core/graph/graph.h"
#include "openmi/idl/proto/engine.pb.h"
#include "openmi/idl/proto/optimizer.pb.h"
using namespace openmi;

namespace openmi {

class ModelParser {
public: 
  static void CreateModelWeightSchema(Graph* g, std::unordered_map<int, std::shared_ptr<proto::internal::ModelWeightSchema> >& schema) {
    for (Node* node: g->source_nodes()) {
      std::string node_name = node->def().name();
      DLOG(INFO) << __FUNCTION__ << " node_name:" << node_name;
      int column_id = -1;
      GetAttr(node->attrs(), "col_id", &column_id);
      if (column_id == -1) {
        CreateNNWeightSchema(node, schema);
      } else {
        CreateColumnWeightSchema(node, column_id, schema);
      }
    }
  }

  static void CreateColumnWeightSchema(Node* node, int column_id, std::unordered_map<int, std::shared_ptr<proto::internal::ModelWeightSchema> >& schema) {
    auto node_name = node->def().name();
    SourceNodeType source_node_type = proto::X;
    GetAttr(node->attrs(), "source_node_type", &source_node_type);
    if (source_node_type != proto::W || node->node_info().node_scope != NS_FORWARD) {
      return;
    }
    
    std::shared_ptr<proto::internal::ModelWeightSchema> column_weight_schema;
    if (schema.find(column_id) != schema.end()) {
      column_weight_schema = schema[column_id];
    } else {
      column_weight_schema = std::make_shared<proto::internal::ModelWeightSchema>();
      schema.insert({column_id, column_weight_schema});
    }
    int embedding_size = 1;
    GetAttr(node->attrs(), "embedding_size", &embedding_size);
    CHECK(embedding_size > 0) << "node '" << node_name 
      << "' has invalid param 'embedding_size' that value should be greater 0";

    // TODO proto::Optimizer
    proto::Optimizer optimizer;
    GetAttr(node->attrs(), "optimizer", &optimizer, false);
    LOG(INFO) << "node: " << node_name << ", optimizer:\n" << optimizer.DebugString();

    // create schema
    column_weight_schema->set_column_id(column_id);
    auto* weight_schema = column_weight_schema->add_weight_schema();
    auto prev_weight_len = column_weight_schema->weight_len();
    auto prev_param_len = column_weight_schema->param_len();
    weight_schema->set_weight_offset(prev_weight_len);
    weight_schema->set_weight_size(embedding_size);
    weight_schema->set_weight_bytes(weight_schema->weight_size() * sizeof(float));
    // TODO 添加optimizer参数
    weight_schema->set_param_bytes(weight_schema->weight_size() * SizeofOptimizer(optimizer.type()));
    column_weight_schema->set_weight_len(prev_weight_len + weight_schema->weight_size());
  }

  static void CreateNNWeightSchema(Node* node, std::unordered_map<int, std::shared_ptr<proto::internal::ModelWeightSchema> >& schema) {
    bool is_label_node = false;
    GetAttr(node->attrs(), "label", &is_label_node);
    if (is_label_node || node->node_info().node_scope != NS_FORWARD) {
      return;
    }

    auto node_name = node->def().name();
    int node_id = 0;
    GetAttr(node->attrs(), "node_id", &node_id);
    CHECK(node_id < 0) 
      << "nn node id must be < 0. but node_id:" << node_id << ", node name:" << node_name;
    
    CHECK(schema.find(node_id) == schema.end())
      << "node id has already exists in schema. node_id:" << node_id << ", node_name:" << node_name;
     
    TensorShape shape;
    GetAttr(node->attrs(), "shape", &shape, false);
    CHECK(shape.Dims() == 2) 
      << "nn node shape.dims != 2. shape:" << shape.DebugString();
    auto dim_2nd = shape.DimSize(1);
    
    proto::Optimizer optimizer;
    GetAttr(node->attrs(), "optimizer", &optimizer, false);

    // create schema
    auto nn_weight_schema = std::make_shared<proto::internal::ModelWeightSchema>();
    schema.insert({node_id, nn_weight_schema});
    nn_weight_schema->set_column_id(node_id);
  
    auto* weight_schema = nn_weight_schema->add_weight_schema();
    weight_schema->set_weight_offset(0);
    weight_schema->set_weight_size(dim_2nd);
    weight_schema->set_weight_bytes(weight_schema->weight_size() * sizeof(float));
    // TODO 添加优化器参数
    weight_schema->set_param_bytes(weight_schema->weight_size() * SizeofOptimizer(optimizer.type()));
    nn_weight_schema->set_weight_len(weight_schema->weight_size());
  }

  static int SizeofOptimizer(openmi::proto::OptimizerType optimizer_type) {
    int bytes = 0;
    switch (optimizer_type) {
      case proto::SGD:
        bytes = 0;
        break;
      case proto::FTRL:
        bytes = 2 * sizeof(float);
        break;
      case proto::ADAGRAD:
        bytes = 1 * sizeof(float);
        break; 
      case proto::ADAM: 
        bytes = 2 * sizeof(float);
        break;  
      default:
        break;
    }
    return bytes;
  }
}; // class ModelParser

} 
#endif // OPENMI_CORE_ENGINE_MODEL_PARSER_H_ 