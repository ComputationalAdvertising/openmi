#include "executor.h"
#include "base/protobuf_op.h"
#include "base/logging.h"

Tensor* GetTensor(Executor& exec, std::string name) {
  Tensor* t = nullptr;
  Status status = exec.session_state_.GetTensor(name, &t);
  CHECK(t != nullptr) << "tensor not found from session state. name: " << name;
  return t;
}

void InitColEmbedding(Tensor** t, std::vector<uint64_t>& batch_dims, const int rank, float v) {
  TensorShape shape(batch_dims);
  (*t)->AllocateTensor(shape);
  (*t)->tensor<float, 2>().setConstant(v);
  DLOG(INFO) << "placeholder variable:\n" << (*t)->tensor<float, 2>();
}

void Iter(Executor& exec, int batch_size) {
  // 1. update source nodes. such as W/X/b
  LOG(INFO) << "================= [placeholder embedding] ================= \n";
  const int rank = 2;
  int column_size = 8;

  std::vector<uint64_t> batch_dims;
  batch_dims.push_back(batch_size);
  batch_dims.push_back(column_size);
  
  Tensor* c1_embed = GetTensor(exec, "c1_embed");
  Tensor* c2_embed = GetTensor(exec, "c2_embed");
  Tensor* c3_embed = GetTensor(exec, "c3_embed");

  InitColEmbedding(&c1_embed, batch_dims, 2, 0.01);
  InitColEmbedding(&c2_embed, batch_dims, 2, -0.02);
  InitColEmbedding(&c3_embed, batch_dims, 2, 0.03);
  
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
  
  label->tensor<float, rank>().setConstant(1);
  label->tensor<float, rank>()(0, 0) = 0;
  DLOG(INFO) << "label:\n" << label->tensor<float, rank>();

  LOG(INFO) << "================= [w_layer1] ================= \n";
  Tensor* w = GetTensor(exec, "w_layer1");
  w->tensor<float, rank>().setConstant(0.03);
  DLOG(INFO)  << "Variable(w_layer1):\n" << w->tensor<float, rank>();

  LOG(INFO) << "================= [b_layer1] ================= \n";
  Tensor* b = GetTensor(exec, "b_layer1");
  b->vec<float>().setConstant(0.00002);
  DLOG(INFO) << "Variable(b_layer1):\n" << b->vec<float>();
  
  // 2. forward & backword
  LOG(INFO) << "================= [exec.run] ================= \n";
  Status s = exec.Run();

  LOG(DEBUG) << "done";

  // 3. push gradients 
}

int main(int argc, char** argv) {
  const char* file = "unittest/conf/wide_and_deep_graph_demo.conf";
  proto::GraphDef gdef;
  if (ProtobufOp::LoadObjectFromPbFile<proto::GraphDef>(file, &gdef) != 0) {
    LOG(ERROR) << "load graph def proto file failed.";
    return -1;
  }

  Executor exec(gdef);
  
  for (Node* node: exec.g_.variable_nodes()) {
    LOG(INFO) << "forward variable node: " << node->def().name();
  }

  for (Node* node: exec.g_.reversed_variable_nodes()) {
    LOG(INFO) << "reversed variable node: " << node->def().name();
  }

  int batch_size = 10;
  //Iter(exec, batch_size);

  batch_size = 3;
  // Iter(exec, batch_size);

  // 1. 获取所有的SourceNode节点
  
  LOG(DEBUG) << "done";
  
  return 0;
}
