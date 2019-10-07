#include "session.h"
#include "base/timer.h"
#include "openmi/core/engine/model_parser.h"

namespace openmi {

Session::Session(): executor_(nullptr) {}

Session::~Session() {}

int Session::Init(proto::GraphDef& gdef) {
  executor_ = std::make_shared<Executor>(gdef);
  if (executor_ == nullptr) {
    LOG(ERROR) << "create executor failed. graph name[" << gdef.name() << "]"; 
    return -1;
  }
  
  if (ParseGraphSourceNode() != 0) {
    LOG(ERROR) << "get graph node from inited executor failed.";
    return -1;
  }

  ModelParser::CreateModelWeightSchema(executor_->GetGraph(), model_weight_schema_);
  LOG(INFO) << "schema_map.size: " << model_weight_schema_.size();

  return 0;
}

void Session::Destroy() {
  executor_->Destroy();
}

int Session::ParseGraphSourceNode() {
  for (Node* node: executor_->GetGraph()->source_nodes()) {
    std::string node_name = node->def().name();
  
    if (node2tensor_.find(node_name) == node2tensor_.end()) {
      Tensor* t = nullptr;
      executor_->GetSessionState()->GetTensor(node_name, &t);
      CHECK(t != nullptr) << "tensor not found from session state. node name: " << node_name;
      node2tensor_.insert({node_name, t});
    }

    std::string related_node_name = node->related_node_name();
    if (node2tensor_.find(related_node_name) == node2tensor_.end()) {
      Tensor* t = nullptr;
      executor_->GetSessionState()->GetTensor(related_node_name, &t);
      CHECK(t != nullptr) << "tensor not found from session state. related node name: " << related_node_name;
      node2tensor_.insert({related_node_name, t});
    }

    int colid = -1;
    GetAttr(node->attrs(), "col_id", &colid);
    if (colid == -1) {
      bool is_label = false;
      GetAttr(node->attrs(), "label", &is_label);
      if (is_label) {
        label_node_.push_back(node_name);
      }
      if (node->node_info().node_scope == NS_FORWARD && !is_label) {
        forward_nn_variable_.push_back(node_name);
        reverse_nn_variable_.push_back(related_node_name);
      }
      continue;
    }

    ColumnNodePtr column_node;
    if (column2node_.find(colid) != column2node_.end()) {
      column_node = column2node_[colid];
    } else {
      column_node = std::make_shared<proto::internal::ColumnNode>();
      if (column_node == nullptr) {
        LOG(ERROR) << "init proto::internal::ColumnNode failed.";
        return -1;
      }
      column2node_.insert({colid, column_node});
      valid_columns_.push_back(colid);
      valid_columns_set_.insert(colid);
      column2keyindex_.insert({colid, std::make_shared<proto::internal::ColumnKeyIndex>()});
    }

    SourceNodeType source_node_type;
    GetAttr(node->attrs(), "source_node_type", &source_node_type);
    switch (source_node_type) {
      case proto::W:
        column_node->add_w(node_name);
        if (node->node_info().node_scope == NS_FORWARD) {
          column_node->add_w_grad(node->related_node_name());

          int embedding_size = 1;
          GetAttr(node->attrs(), "embedding_size", &embedding_size);
          auto* weight_schema = column_node->mutable_weight_schema();
          weight_schema->set_column_id(colid);

          auto* weight_offset = weight_schema->add_weight_offset();
          weight_offset->set_weight_offset(weight_schema->total_weight_size());
          weight_offset->set_weight_size(embedding_size);

          weight_schema->set_total_weight_size(weight_schema->total_weight_size() + embedding_size);
          LOG(INFO) << "column node:\n" << column_node->DebugString();
        }
        break;
      case proto::X: 
        column_node->set_x(node_name);
        break;
      case proto::OFFSET: 
        column_node->set_row_offset(node_name);
        break;
      default:
        break;
    }
  } // end for

  size_t forward_nn_size = forward_nn_variable_.size();
  size_t reverse_nn_size = reverse_nn_variable_.size();
  if (forward_nn_size != reverse_nn_size) {
    LOG(ERROR) << "size of variable between forward and reverse not match. "
      << "forward nn variable size:" << forward_nn_size 
      << ", reverse nn variable size:" << reverse_nn_size;
    return -1; 
  }

  // nn weight sign that used to graph cache
  GetNNFids();

  if (CheckColumnNode() != 0) {
    LOG(INFO) << "column node from graph is illegal.";
    return -1;
  }

  return 0;
}

int Session::CheckColumnNode() {
  auto it = column2node_.begin();
  while (it != column2node_.end()) {
    auto column_node = it->second;
    if (column_node->x() == "") {
      LOG(ERROR) << "column node 'x' name is empty. column node:\n" << column_node->DebugString();
      return -1;
    }
    if (column_node->row_offset() == "") {
      LOG(ERROR) << "column node 'row_offset' name is empty. column node:\n" << column_node->DebugString();
      return -1;
    }
    if (column_node->w_size() == 0) {
      LOG(ERROR) << "column node 'w' name is empty. column node:\n" << column_node->DebugString();
      return -1;
    }
    it++;
  }
  return 0;
}

int Session::Run(InstancesPtr& batch, bool is_training) {
  batch_ = batch;
  Timer time;
  if (Pull() != 0) {
    LOG(ERROR) << "pull phase failed.";
    return -1;
  }
  LOG(INFO) << "pull phase time: " << time.Elapsed();

  if (FeedSourceNode(is_training) != 0) {
    LOG(ERROR) << "feed column node failed.";
    return -1;
  }
  LOG(INFO) << "feed column node time: " << time.Elapsed();

  executor_->Run();
  LOG(INFO) << "exec forward&&backward time: " << time.Elapsed();

  if (is_training) {
    if (GetGradient() != 0) {
      LOG(ERROR)  << "get gradient failed.";
      return -1;
    }
    LOG(INFO) << "get gradient phase time: " << time.Elapsed();
  
    if (Push() != 0) {
      LOG(ERROR) << "push failed.";
      return -1;
    }
    LOG(INFO) << "push phase time: " << time.Elapsed();
  }
  return 0; 
}

int Session::Pull() {
  ps_accessor_.ClearPull();
  fid_set_.clear();
  // batch fids
  for (int i = 0; i < batch_->instance_size(); ++i) {
    auto ins = batch_->instance(i);
    for (int j = 0; j < ins.feature_size(); ++j) {
      auto fid = ins.feature(j).fid();
      auto colid = ins.feature(j).colid();
      if (valid_columns_set_.find(colid) == valid_columns_set_.end()) {
        continue;
      }
      if (fid_set_.find(fid) != fid_set_.end()) {
        continue;
      }
      fid_set_.insert(fid);
      ps_accessor_.AddPullFid(fid, colid);
    }
  }

  // nn fids cache
  time_t now = std::time(nullptr);
  if (now - last_update_time_ > graph_cache_time) {
    update_graph_cache_ = true;
    last_update_time_ = now;
  } else {
    update_graph_cache_ = false;
  }

  if (update_graph_cache_) {
    for (auto fid: nn_fids_) {
      ps_accessor_.AddPullFid(fid);
    }
  }

  ps_accessor_.PreparePull();
  ps_accessor_.Pull();

  return 0;
}

int Session::Push() {
  // TODO
  return 0;
}

int Session::FeedSourceNode(bool is_training) {
  // feed column node
  // TODO parallel
  for (auto i = 0; i < valid_columns_.size(); ++i) {
    auto colid = valid_columns_[i];
    if (FeedColumnNode(colid) != 0) {
      LOG(ERROR) << "feed column node failed. column id:" << valid_columns_[i];
      return -1;
    }
  }
  // feed nn node 
  if (update_graph_cache_) {
    // TODO parallel
    for (auto idx = 0; idx < forward_nn_variable_.size(); ++idx) {
      auto nn_node_name = forward_nn_variable_[idx];
      if (FeedNNNode(idx, nn_node_name) != 0) {
        LOG(ERROR) << "feed nn node failed. nn_node_name:" << nn_node_name;
        return -1;
      }
    }
  }

  if (is_training) {
    FeedLabelNode();
  }
  return 0;
}

int Session::FeedLabelNode() {
  for (size_t i = 0; i < label_node_.size(); ++i) {
    auto label_node_name = label_node_[i];
    auto* label = node2tensor_[label_node_name];
    int batch_size = batch_->instance_size();
    // TODO shape需要支持多标签
    std::string label_shape_s = std::to_string(batch_size) + ",1";
    TensorShape label_shape(label_shape_s);
    LOG(INFO) << "label shape:" << label_shape.DebugString();
    label->AllocateTensor(label_shape);
    LOG(INFO) << "allocate tensor done";
    for (int row_idx = 0; row_idx < batch_size; ++row_idx) {
      auto ins = batch_->instance(row_idx);
      CHECK(ins.label().labels_size() > 0);
      auto Y = ins.label().labels(0);
      LOG(INFO) << "Y:" << Y;
      label->tensor<float, 2>()(row_idx, 0) = Y;
    }
    DLOG(INFO) << "label_node_name:" << label_node_name << ", tensor:\n" << label->tensor<float, 2>();
  }
  return 0;
}

//int continuous_feature_size = ins.continuous_feature_size();
//如何优雅的填充连续特征？
// int Session::FeedColumnNode(InstancesPtr& batch, uint32_t colid) {
//   return 0;
// }

int Session::FeedColumnNode(uint32_t colid) {
  CHECK(column2node_.find(colid) != column2node_.end());
  auto column_node = column2node_[colid];
  auto row_offset_node_name = column_node->row_offset();

  auto column_key_index = column2keyindex_.at(colid);
  column_key_index->Clear();

  int batch_size = batch_->instance_size();
  LOG(INFO) << "batch size:" << batch_size;
  Tensor* tensor = node2tensor_[row_offset_node_name];
  if (!tensor->IsInitialized() || batch_size != tensor->shape().DimSize(0)) {
    std::string shape_str = std::to_string(batch_size);
    TensorShape shape(shape_str);
    tensor->AllocateTensor(shape);
  }

  // consider multi hot encoding
  std::vector<float> x_values;   

  uint32_t offset = 0;
  for (int i = 0; i < batch_size; ++i) {
    auto ins = batch_->instance(i);

    int count = 0;
    for (int j = 0; j < ins.feature_size(); ++j) {
      auto fea = ins.feature(j);
      if (fea.colid() != colid) {
        continue;
      }
      count += 1;
      x_values.push_back(fea.weight());
      auto fid = fea.fid();
      column_key_index->add_keys(fid);
    }

    // weight default value(0)
    if (count == 0) {  
      count = 1;
      x_values.push_back(0);
      column_key_index->add_keys(0);
    }

    offset += count;
    tensor->vec<uint32_t>()(i) = offset;
  }
  DLOG(INFO) << "row_offset:\n" << tensor->vec<uint32_t>();

  FeedXNode(column_node, x_values);

  FeedWNode(column_node, column_key_index);

  return 0;
}
// for sparse data
int Session::FeedXNode(ColumnNodePtr& column_node, std::vector<float>& values) {
  auto node_name = column_node->x();
  Tensor* tensor = node2tensor_[node_name];
  if (!tensor->IsInitialized() || values.size() != tensor->shape().DimSize(0)) {
    std::string shape_str = std::to_string(values.size()) + ",1";
    TensorShape shape(shape_str);
    tensor->AllocateTensor(shape);
  }
  auto x = tensor->tensor<float, 2>();
  
  // TODO 使用指针和size初始化tensor。 这里暂时使用原始方法；
  for (int idx = 0; idx < values.size(); ++idx) {
    x(idx, 0) = values[idx];
  }
  DLOG(INFO) << "feed x node[" << node_name << "]:\n" << x;
  return 0;
}

int Session::FeedWNode(ColumnNodePtr& column_node, ColumnKeyIndexPtr& fids) {
  // consider multiply weight node respond to one x source node
  for (int k = 0; k < column_node->w_size(); ++k) {
    auto node_name = column_node->w(k);
    auto weight_offset = column_node->weight_schema().weight_offset(k);
    auto offset = weight_offset.weight_offset();
    auto size = weight_offset.weight_size();
    Tensor* tensor = node2tensor_[node_name];
    std::string shape_str = std::to_string(fids->keys_size()) + "," + std::to_string(size);
    TensorShape shape(shape_str);
    if (!tensor->IsInitialized() || tensor->shape() != shape) {
      tensor->AllocateTensor(shape);
    }

    auto w = tensor->tensor<float, 2>();
    w.setZero();

    auto fid2paramdata = ps_accessor_.GetFid2ParamData();
    for (int i = 0; i < fids->keys_size(); ++i) {
      auto fid = fids->keys(i);
      if (fid == 0 || fid2paramdata.find(fid) == fid2paramdata.end()) {
        continue;
      }

      auto val_list = fid2paramdata[fid];
      CHECK(val_list.val_size() >= offset + size);
      // TODO 使用指针和size初始化tensor的每一行，而不是element级别赋值
      for (int j = 0; j < size; ++j) {
        w(i, j) = val_list.val(offset + j);
      }
    }
    DLOG(INFO) << "feed w node[" << node_name << "]:\n" << w;
  }
  return 0;
}

uint64_t Session::NNFid(int node_idx, int row_idx) {
  return GetFid(node_idx + 1, row_idx + 1);
}

void Session::GetNNFids() {
  nn_fids_.clear();
  for (auto i = 0; i < forward_nn_variable_.size(); ++i) {
    auto node_name = forward_nn_variable_.at(i);
    CHECK(node2tensor_.find(node_name) != node2tensor_.end());
    Tensor* tensor = node2tensor_[node_name];
    CHECK(tensor->IsInitialized()) << "graph weight has not initialized.";
    CHECK(tensor->shape().Dims() == 2);
    auto dim_1st = tensor->shape().DimSize(0);
    for (auto row_idx = 0; row_idx < dim_1st; ++row_idx) {
      auto fid = NNFid(i, row_idx);
      nn_fids_.push_back(fid);
    }
  }
}

int Session::FeedNNNode(int node_idx, std::string& node_name) {
  auto model_param_data = ps_accessor_.GetFid2ParamData();

  if (node2tensor_.find(node_name) == node2tensor_.end()) {
    LOG(ERROR) << "nn node '" << node_name << "' not in session state.";
    return -1;
  }
  Tensor* nn = node2tensor_[node_name];
  // by first dim (default by row splitted)
  auto first_dim_size = nn->shape().DimSize(0);
  auto second_dim_size = nn->shape().DimSize(1);
  // row index
  for (auto idx = 0; idx < first_dim_size; ++idx) {
    uint64_t fid = NNFid(node_idx, idx);
    if (model_param_data.find(fid) != model_param_data.end()) {
      auto val_list = model_param_data[fid];
      CHECK(val_list.val_size() == (int)second_dim_size) << "nn cols size not match";
      if (val_list.val_size() != (int)second_dim_size) {
        LOG(ERROR) << "shape not match between nn tensor second dim and val list. "
                   << second_dim_size << " vs " << val_list.val_size();
        return -1;
      }
      for (int j = 0; j < val_list.val_size(); ++j) {
        nn->tensor<float, 2>()(idx, j) = val_list.val(j);
      }
    } else {
      // TODO 需要优化初始化方式
      for (int j = 0; j < second_dim_size; ++j) {
        //nn->tensor<float, 2>()(idx, j) = 0;
        nn->tensor<float, 2>()(idx, j) = 1;
      }
    }
  } // for each row
  LOG(INFO) << "forward nn node [" << node_name << "], tensor:\n" << nn->tensor<float, 2>();
  return 0;
}

int Session::GetGradient() {
  std::unordered_map<uint64_t, ValListPtr> fid2grad;
  // get column node gradient
  // TODO parallel
  for (auto i = 0; i < valid_columns_.size(); ++i) {
    auto colid = valid_columns_[i];
    if (GetColumnGradient(colid, fid2grad) != 0) {
      LOG(ERROR) << "feed column node failed. column id:" << valid_columns_[i];
      return -1;
    }
  }
  // get nn node gradient
  // TODO parallel
  for (auto idx = 0; idx < reverse_nn_variable_.size(); ++idx) {
    auto nn_node_name = reverse_nn_variable_[idx];
    if (GetNNGradient(idx, nn_node_name, fid2grad) != 0) {
      LOG(ERROR) << "feed nn gradient failed. node_name:" << nn_node_name;
      return -1;
    }
  }

  // TODO prepare push
  // fid2grad -> proto::internal::ModelParamData

  fid2grad.clear();
  return 0;
}

int Session::GetColumnGradient(int colid, std::unordered_map<uint64_t, ValListPtr>& fid2grad) {
  auto column_key_index = column2keyindex_.at(colid);
  auto column_node = column2node_[colid];
  auto total_weight_size = column_node->weight_schema().total_weight_size();

  // multiply weight node
  for (int k = 0; k < column_node->w_size(); ++k) {
    auto w_node_name = column_node->w_grad(k);
    CHECK(node2tensor_.find(w_node_name) != node2tensor_.end());
    auto* tensor = node2tensor_[w_node_name];
    CHECK(tensor->IsInitialized());
    auto grad = tensor->tensor<float, 2>();
    auto row_size = tensor->shape().DimSize(0);
    CHECK(row_size == column_key_index->keys_size());
    
    auto weight_offset = column_node->weight_schema().weight_offset(k);
    auto offset = weight_offset.weight_offset();
    auto size = weight_offset.weight_size();

    for (int row_idx = 0; row_idx < row_size; ++row_idx) {
      auto fid = column_key_index->keys(row_idx);
      
      ValListPtr grad_list;
      if (fid2grad.find(fid) != fid2grad.end()) {
        grad_list = fid2grad[fid];
      } else {
        grad_list = std::make_shared<proto::internal::ValList>();
        grad_list->mutable_val()->Resize(total_weight_size, 0);
        fid2grad.insert({fid, grad_list});
      }

      if (row_idx == 0) {
        CHECK(size == tensor->shape().DimSize(1)) 
          << "size:" << size << ", dim(1):" << tensor->shape().DimSize(1);
      }
      
      for (int j = 0; j < size; ++j) {
        auto offset1 = offset + j;
        grad_list->set_val(offset1, grad_list->val(offset1) + grad(row_idx, j));
      }
    } // row index
  } // multi weight node
  LOG(INFO) << "done";
  return 0;
}

int Session::GetNNGradient(int idx, std::string& node_name, std::unordered_map<uint64_t, ValListPtr>& fid2grad) {
  LOG(INFO) << "reversed nn:" << node_name;
  CHECK(node2tensor_.find(node_name) != node2tensor_.end());
  
  auto* nn = node2tensor_[node_name];
  auto grad = nn->tensor<float, 2>();

  LOG(INFO) << ", nn_grad_tensor:\n" << grad;
  // by first dim (default by row splitted)
  auto dim_1st = nn->shape().DimSize(0);
  auto dim_2nd = nn->shape().DimSize(1);
  for (int row_idx = 0; row_idx < dim_1st; ++row_idx) {
    auto fid = NNFid(idx, row_idx);

    ValListPtr grad_list;
    if (fid2grad.find(fid) != fid2grad.end()) {
      grad_list = fid2grad[fid];
    } else {
      grad_list = std::make_shared<proto::internal::ValList>();
      grad_list->mutable_val()->Resize(dim_2nd, 0);
      fid2grad.insert({fid, grad_list});
    }

    // 优化性能
    for (int j = 0; j < dim_2nd; ++j) {
      grad_list->set_val(j, grad_list->val(j) + grad(row_idx, j));
    }
  }

  return 0;
}

} // namespace openmi