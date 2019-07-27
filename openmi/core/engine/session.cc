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
  
  if (GetGraphNode() != 0) {
    LOG(ERROR) << "get graph node from inited executor failed.";
    return -1;
  }
  return 0;
}

void Session::Destroy() {
  executor_->Destroy();
}

int Session::GetGraphNode() {
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
  return 0;
}

}