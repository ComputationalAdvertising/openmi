#include "executor.h"
#include "base/protobuf_op.h"
#include "base/logging.h"

Tensor* GetTensor(Executor& exec, std::string name) {
  Tensor* t = nullptr;
  Status status = exec.session_state_.GetTensor(name, &t);
  return t;
}

template <typename T>
typename MTypes<T>::Matrix ToEigenMatrix(Tensor& tensor) {
  auto m = tensor.matrix<T>();
  return Eigen::Map<typename MTypes<T>::Matrix>(
    m.data(), m.dimension(0), m.dimension(1));
}

// RowMajor
template <typename T>
typename MTypes<T>::Matrix ToEigenVector(Tensor& tensor) {
  auto v = tensor.vec<T>();
  return Eigen::Map<typename MTypes<T>::Matrix>(
    v.data(), 1, v.dimension(0));
}

int main(int argc, char** argv) {
  const char* file = "unittest/conf/graph_demo.conf";
  proto::GraphDef gdef;
  if (ProtobufOp::LoadObjectFromPbFile<proto::GraphDef>(file, &gdef) != 0) {
    LOG(ERROR) << "load graph def proto file failed.";
    return -1;
  }

  Executor exec(gdef);

  const int rank = 2;

  LOG(INFO) << "================= [x] ================= \n";
  Tensor* x = GetTensor(exec, "x");
  TensorShape shape("6,8");
  x->set_shape(shape);
  x->Init();
  x->tensor<float, rank>().setConstant(0.2);
  DLOG(INFO)  << "Variable(x):\n" << x->tensor<float, 2>(); 
  //auto matrix_x = ToEigenMatrix<float>(*x);
  auto matrix_x = x->ToEigenMatrix<float>();
  DLOG(INFO) << "Matrix[x]:\n" << matrix_x << "\trows:" << matrix_x.rows() << ", cols:" << matrix_x.cols();
  
  LOG(INFO) << "================= [label] ================= \n";
  Tensor* label = GetTensor(exec, "label");
  TensorShape lshape("6,1");
  label->AllocateTensor(lshape);
  label->tensor<float, rank>().setConstant(1);
  label->tensor<float, rank>()(0, 0) = 0;
  label->tensor<float, rank>()(2, 0) = 0;
  label->tensor<float, rank>()(4, 0) = 0;

  LOG(INFO) << "================= [w] ================= \n";
  Tensor* w = GetTensor(exec, "w");
  w->tensor<float, rank>().setConstant(0.3);
  DLOG(INFO)  << "Variable(w):\n" << w->tensor<float, rank>();

  auto matrix_w = ToEigenMatrix<float>(*w);
  DLOG(INFO) << "matrix_w:\n" << matrix_w << "\trows:" << matrix_w.rows() << ", cols:" << matrix_w.cols();

  LOG(INFO) << "================= [b] ================= \n";
  Tensor* b = GetTensor(exec, "b");
  LOG(INFO) << "TensorShape(b): " << b->shape().DebugString();
  b->vec<float>().setConstant(0.002);
  DLOG(INFO) << "Variable(b):\n" << b->vec<float>();
  
  LOG(INFO) << "================= [exec.run] ================= \n";
  // y = 0.991998
  Status s = exec.Run();

  LOG(DEBUG) << "done";
  
  /*
  auto vec_b = ToEigenVector<float>(*b);
  LOG(INFO) << "vec_b:\n" << vec_b << "\nrows:" << vec_b.rows() << ", cols:" << vec_b.cols();
  auto wx = matrix_x * matrix_w.transpose();
  LOG(INFO) << "wx:\n" << wx;
  auto wxb = wx.rowwise() + vec_b.row(0);
  LOG(INFO) << "wxb:\n" << wxb << "\nrows: " << wxb.rows() << ", cols:" << wxb.cols();
  */
  
  return 0;
}
