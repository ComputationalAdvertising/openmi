#include "session.h"

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
        reverse_nn_variable_.push_back(node->related_node_name());
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
    }

    SourceNodeType source_node_type;
    GetAttr(node->attrs(), "source_node_type", &source_node_type);
    switch (source_node_type) {
      case proto::W:
        if (node->node_info().node_scope == NS_FORWARD) {
          column_node->add_w_grad(node->related_node_name());
        }
        column_node->add_w(node_name);
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

int Session::FeedSourceNode(InstancesPtr& batch) {
  LOG(INFO) << "valid column size:" << valid_columns_.size();
  for (auto i = 0; i < valid_columns_.size(); ++i) {
    auto colid = valid_columns_[i];
    if (FeedColumnNode(batch, colid, 8) != 0) {
      LOG(ERROR) << "feed column node failed. column id:" << valid_columns_[i];
      return -1;
    }
  }
  return 0;
}

//int continuous_feature_size = ins.continuous_feature_size();
//如何优雅的填充连续特征？
int Session::FeedColumnNode(InstancesPtr& batch, uint32_t colid) {
  return 0;
}

int Session::FeedColumnNode(InstancesPtr& batch, uint32_t colid, int embedding_size) {
  LOG(INFO) << "feed colid: " << colid; 
  auto column_node = column2node_[colid];
  auto x_node_name = column_node->x();
  auto w_node_name = column_node->w(0);
  auto row_offset_node_name = column_node->row_offset();

  LOG(INFO) << "x_node_name:" << x_node_name
            << ", w_node_name:" << w_node_name 
            << ", row_offset_node_name:" << row_offset_node_name;

  int batch_size = batch->instance_size();
  Tensor* row_offset = node2tensor_[row_offset_node_name];
  std::string offset_shape_s = std::to_string(batch_size);
  TensorShape offset_shape(offset_shape_s);
  row_offset->AllocateTensor(offset_shape);

  std::vector<float> x_values;
  std::vector<std::vector<float>> w_values;

  int32_t offset = 0;
  for (int i = 0; i < batch_size; ++i) {
    auto ins = batch->instance(i);
    int32_t count = 0;
    int discrete_feature_size = ins.feature_size();
    for (int j = 0; j < discrete_feature_size; ++j) {
      auto fea = ins.feature(j);
      if (fea.colid() != colid) {
        continue;
      }
      count += 1;
      x_values.push_back(fea.weight());
      auto fid = fea.fid();
      // TODO fill W
    }
    // fill default 0
    if (count == 0) {  
      count = 1;
      x_values.push_back(0);
    }

    offset += count;
    row_offset->vec<uint32_t>()(i) = offset;
  }

  LOG(INFO) << "row_offset:\n" << row_offset->vec<uint32_t>();

  Tensor* x = node2tensor_[x_node_name];
  std::string x_shape_s = std::to_string(x_values.size()) + ",1";
  TensorShape x_shape(x_shape_s);
  x->AllocateTensor(x_shape);
  // TODO 使用指针和size初始化tensor。 这里暂时使用原始方法；
  auto x_value_size = x_values.size();
  for (int idx = 0; idx < x_value_size; ++idx) {
    x->tensor<float, 2>()(idx, 0) = x_values[idx];
  }

  LOG(INFO) << "x:\n" << x->tensor<float, 2>();  

  return 0;
}

} // namespace openmi