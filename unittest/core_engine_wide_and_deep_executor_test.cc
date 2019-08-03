#include "executor.h"
#include "session.h"
#include "attr_value_utils.h"
#include "openmi/idl/proto/engine.pb.h"
#include "base/protobuf_op.h"
#include "base/logging.h"
#include <unordered_set>

using namespace openmi;

Tensor* GetTensor(Executor& exec, std::string name) {
  Tensor* t = nullptr;
  Status status = exec.GetSessionState()->GetTensor(name, &t);
  CHECK(t != nullptr) << "tensor not found from session state. name: " << name;
  return t;
}

void InitColEmbedding(Tensor** t, std::vector<uint64_t>& batch_dims, const int rank, float v) {
  TensorShape shape(batch_dims);
  (*t)->AllocateTensor(shape);
  (*t)->tensor<float, 2>().setConstant(v);
  DLOG(INFO) << "placeholder variable:\n" << (*t)->tensor<float, 2>();
}

void FillFeatureWeight(Tensor** t, std::vector<uint64_t>& batch_dims, int colid, float v) {
  TensorShape shape(batch_dims);
  (*t)->AllocateTensor(shape);
  (*t)->tensor<float, 2>().setConstant(v);
  LOG(INFO) << __FUNCTION__ << " colid[" << colid << "] feature weight:\n" << (*t)->tensor<float, 2>();
}

void FillFeatureValue(Tensor** t, std::vector<uint64_t>& batch_dims, int colid, float v) {
  TensorShape shape(batch_dims);
  (*t)->AllocateTensor(shape);
  (*t)->tensor<float, 2>().setConstant(v);
  LOG(INFO) << __FUNCTION__ << " colid[" << colid << "] feature value:\n" << (*t)->tensor<float, 2>();
}

void FillRowOffset(Tensor** t, std::vector<uint64_t>& batch_dims, int colid, int max_offset) {
  typedef int32_t T;
  TensorShape shape(batch_dims);
  (*t)->AllocateTensor(shape);
  (*t)->vec<T>().setConstant(1);
  for (int i = 0; i < batch_dims[0]; ++i) {
    int offset = (i == batch_dims[0] - 1) ? max_offset : i+1;
    (*t)->vec<T>()(i) = offset;
  }
  LOG(INFO) << __FUNCTION__ << " colid[" << colid << "], row offset:\n" << (*t)->vec<T>();
}

void Iter(Executor& exec, int batch_size) {
  LOG(INFO) << "================= [placeholder embedding] weigth/value/offset ================= \n";
  LOG(INFO) << "batch_size[" << batch_size << "]. update feature weight/value/offset.";
  int value_size = batch_size * 2;
  int embedding_size = 8;
  std::vector<uint64_t> weight_dims;
  weight_dims.push_back(value_size);
  weight_dims.push_back(embedding_size);
  
  Tensor* embed_c1 = GetTensor(exec, "embed_c1");
  Tensor* embed_c2 = GetTensor(exec, "embed_c2");
  FillFeatureWeight(&embed_c1, weight_dims, 1, 0.1);
  FillFeatureWeight(&embed_c2, weight_dims, 2, 0.2);

  weight_dims[1] = 1;
  Tensor* x_c1 = GetTensor(exec, "x_c1");
  Tensor* x_c2 = GetTensor(exec, "x_c2");
  FillFeatureValue(&x_c1, weight_dims, 1, 0.1);
  FillFeatureValue(&x_c2, weight_dims, 2, 0.2);

  std::vector<uint64_t> offset_dims;
  offset_dims.push_back(batch_size);
  //offset_dims.push_back(1);
  Tensor* row_offset_c1 = GetTensor(exec, "row_offset_c1");
  Tensor* row_offset_c2 = GetTensor(exec, "row_offset_c2");
  FillRowOffset(&row_offset_c1, offset_dims, 1, value_size);
  FillRowOffset(&row_offset_c2, offset_dims, 2, value_size);

  LOG(INFO) << "================= [placeholder linear] ================= \n";
  std::vector<uint64_t> linear_batch_dims;
  linear_batch_dims.push_back(batch_size);
  linear_batch_dims.push_back(1L);

  Tensor* c1_linear = GetTensor(exec, "c1_linear");
  Tensor* c3_linear = GetTensor(exec, "c3_linear");

  InitColEmbedding(&c1_linear, linear_batch_dims, 2, 0.0001);
  InitColEmbedding(&c3_linear, linear_batch_dims, 2, 0.0003);
  
  LOG(INFO) << "================= [label] ================= \n";
  int num_label_dim = 1;
  Tensor* label = GetTensor(exec, "label");
  
  std::vector<uint64_t> label_dims;
  label_dims.push_back(batch_size);
  label_dims.push_back(num_label_dim);

  TensorShape lshape(label_dims);
  label->AllocateTensor(lshape);
  
  label->tensor<float, 2>().setConstant(1);
  label->tensor<float, 2>()(0, 0) = 0;
  DLOG(INFO) << "label:\n" << label->tensor<float, 2>();

  LOG(INFO) << "================= [w_layer1] ================= \n";
  Tensor* w = GetTensor(exec, "w_layer1");
  w->tensor<float, 2>().setConstant(0.03);
  DLOG(INFO)  << "Variable(w_layer1):\n" << w->tensor<float, 2>();

  LOG(INFO) << "================= [b_layer1] ================= \n";
  Tensor* b = GetTensor(exec, "b_layer1");
  b->vec<float>().setConstant(0.00002);
  DLOG(INFO) << "Variable(b_layer1):\n" << b->vec<float>();
  
  // 2. forward & backword
  LOG(INFO) << "================= [exec.run] ================= \n";
  Status s = exec.Run();

  LOG(INFO) << "================= [after run. get grad info] ================= \n";
  for (Node* node: exec.GetGraph()->reversed_variable_nodes()) {
    std::string grad_node_name = node->def().name();
    std::string related_node_name = node->related_node_name();
    Tensor* grad = GetTensor(exec, grad_node_name);
    LOG(INFO) << "grad node:" << grad_node_name << ", related_node_name:" << related_node_name;
    LOG(INFO) << "value:\n" << grad->matrix<float>();
    LOG(INFO) << "its shape: " << grad->shape().DebugString();
  }

  LOG(DEBUG) << "done";
}

void VariableGradTest(std::unordered_map<std::string, Tensor*>& node2tensor_) {
  LOG(INFO) << "all valid source node.";
  for (auto it = node2tensor_.begin(); it != node2tensor_.end(); it++) {
    LOG(INFO) << "node: " + it->first << ", shape: " << it->second->shape().DebugString();
  }
}

int main(int argc, char** argv) {
  const char* file = "unittest/conf/wide_and_deep_graph_demo.conf";
  proto::GraphDef gdef;
  if (ProtobufOp::LoadObjectFromPbFile<proto::GraphDef>(file, &gdef) != 0) {
    LOG(ERROR) << "load graph def proto file failed.";
    return -1;
  }

  Session sess;
  if (sess.Init(gdef) != 0) {
    LOG(ERROR) << "session init failed.";
    return -1;
  }

  Executor* exec = sess.GetExecutor().get();
  
  // int batch_size = 5;
  // Iter(*exec, batch_size);
  // Iter(*exec, batch_size*2);

  InstancesPtr instances = std::make_shared<proto::Instances>();
  for (int i = 0; i < 10; ++i) {
    auto ins1 = instances->add_instance();
    int f_size = 10;
    for (int j = 0; j < f_size; ++j) {
      auto f = ins1->add_feature();
      f->set_colid(j%3 + 1);
      f->set_fid((j+1)*1000);
      f->set_weight((j+1)*0.1);
    }
  }
  sess.FeedSourceNode(instances);
  
  // 1. 获取所有的SourceNode节点 (理应包括所有的reversed variable node)
  
  LOG(DEBUG) << "done";
  
  return 0;
}
