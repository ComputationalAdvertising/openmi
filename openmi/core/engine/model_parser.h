#ifndef OPENMI_CORE_ENGINE_MODEL_PARSER_H_
#define OPENMI_CORE_ENGINE_MODEL_PARSER_H_

#include "openmi/core/graph/graph.h"
#include "openmi/idl/proto/engine.pb.h"
#include "openmi/idl/proto/optimizer.pb.h"
using namespace openmi;

namespace openmi {

class ModelParser {
public: 
  static int CreateModelWeightSchema(Graph* g, std::unordered_map<int, std::shared_ptr<proto::internal::ModelWeightSchema> >& schema) {
    for (Node* node: g->source_nodes()) {
      std::string node_name = node->def().name();
      DLOG(INFO) << __FUNCTION__ << " node_name:" << node_name;
      int column_id = -1;
      GetAttr(node->attrs(), "col_id", &column_id);
      if (column_id == -1) {
        if (CreateNNWeightSchema(node, schema) != 0) {
          LOG(ERROR) << __FUNCTION__ << " Error: create nn weight schema.";
          return -1;
        }
      } else {
        if (CreateColumnWeightSchema(node, column_id, schema) != 0) {
          LOG(ERROR) << __FUNCTION__ << " Error: create column weight schema.";
          return -1;
        }
      }
    }
    return 0;
  }

  /**
   * \brief column weight schema 
   *    weight: w1,v1,v2,...,v8
   *    optimizer: {wz1},{vz1,vn1},{vz2,vn2},...,{vz8,vn8}
   */
  static int CreateColumnWeightSchema(Node* node, int column_id, std::unordered_map<int, std::shared_ptr<proto::internal::ModelWeightSchema> >& schema) {
    auto node_name = node->def().name();
    SourceNodeType source_node_type = proto::X;
    GetAttr(node->attrs(), "source_node_type", &source_node_type);
    if (source_node_type != proto::W || node->node_info().node_scope != NS_FORWARD) {
      return 0;
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
    if (embedding_size <= 0) {
      LOG(ERROR) << "'" << node_name 
                 << "' node has invalid param 'embedding_size' that value should be great 0";
      return -1;
    }

    // TODO proto::Optimizer
    proto::Optimizer optimizer;
    GetAttr(node->attrs(), "optimizer", &optimizer, false);
    LOG(INFO) << "node: " << node_name << ", optimizer:\n" << optimizer.DebugString();

    // create schema
    column_weight_schema->set_column_id(column_id);
    auto* weight_schema = column_weight_schema->add_weight_schema();
    auto prev_total_weight_size = column_weight_schema->total_weight_size();
    auto prev_total_weight_bytes = column_weight_schema->total_weight_bytes();
    auto prev_total_weight_optimizer_bytes = column_weight_schema->total_weight_optimizer_bytes();
    //weight_schema->set_bytes_offset(prev_total_weight_bytes + prev_total_param_bytes);

    weight_schema->set_weight_offset(prev_total_weight_size);
    weight_schema->set_weight_size(embedding_size);
    weight_schema->set_weight_bytes_offset(prev_total_weight_bytes);
    weight_schema->set_weight_bytes(weight_schema->weight_size() * sizeof(float));
    weight_schema->set_weight_optimizer_bytes_offset(prev_total_weight_optimizer_bytes);
    weight_schema->set_weight_optimizer_bytes(weight_schema->weight_size() * SizeofOptimizer(optimizer.type()));
    
    column_weight_schema->set_total_weight_size(prev_total_weight_size + weight_schema->weight_size());
    column_weight_schema->set_total_weight_bytes(prev_total_weight_bytes + weight_schema->weight_bytes());
    column_weight_schema->set_total_weight_optimizer_bytes(prev_total_weight_optimizer_bytes + weight_schema->weight_optimizer_bytes());

    return 0;
  }

  /**
   * \brief nn weight structure: w1,w2,...,w_n | m1,m2,...,m_n | v1,v2,...,v_n
   */
  static int CreateNNWeightSchema(Node* node, std::unordered_map<int, std::shared_ptr<proto::internal::ModelWeightSchema> >& schema) {
    bool is_label_node = false;
    GetAttr(node->attrs(), "label", &is_label_node);
    if (is_label_node || node->node_info().node_scope != NS_FORWARD) {
      return 0;
    }

    auto node_name = node->def().name();
    int node_id = 0;
    GetAttr(node->attrs(), "node_id", &node_id);
    if (node_id >= 0) {
      LOG(ERROR) << "nn node id must be < 0. but node_id:" << node_id << ", node name:" << node_name;
      return -1;
    }
    
    if (schema.find(node_id) != schema.end()) {
      LOG(ERROR) << "node id has already exists in schema. node_id:" << node_id << ", node_name:" << node_name;
      return -1;
    }
     
    TensorShape shape;
    GetAttr(node->attrs(), "shape", &shape, false);
    if (shape.Dims() != 2) {
      LOG(ERROR) << "nn node shape.dims != 2. shape:" << shape.DebugString();
      return -1;
    }
      
    auto dim_2nd = shape.DimSize(1);
    proto::Optimizer optimizer;
    GetAttr(node->attrs(), "optimizer", &optimizer, false);

    // create schema
    auto nn_weight_schema = std::make_shared<proto::internal::ModelWeightSchema>();
    schema.insert({node_id, nn_weight_schema});
    nn_weight_schema->set_column_id(node_id);
  
    auto* weight_schema = nn_weight_schema->add_weight_schema();

    // TODO not only 2nd dim, but extra dims
    weight_schema->set_weight_offset(0);
    weight_schema->set_weight_size(dim_2nd);
    weight_schema->set_weight_bytes_offset(0);
    weight_schema->set_weight_bytes(weight_schema->weight_size() * sizeof(float));
    weight_schema->set_weight_optimizer_bytes_offset(0);
    weight_schema->set_weight_optimizer_bytes(weight_schema->weight_size() * SizeofOptimizer(optimizer.type()));

    nn_weight_schema->set_total_weight_size(nn_weight_schema->total_weight_size() + weight_schema->weight_size());
    nn_weight_schema->set_total_weight_bytes(nn_weight_schema->total_weight_bytes() + weight_schema->weight_bytes());
    nn_weight_schema->set_total_weight_optimizer_bytes(nn_weight_schema->total_weight_optimizer_bytes() + weight_schema->weight_optimizer_bytes());

    return 0;
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