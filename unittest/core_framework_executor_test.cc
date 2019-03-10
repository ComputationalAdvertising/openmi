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

  Tensor* x = GetTensor(exec, "x");
  TensorShape shape("6,8");
  x->set_shape(shape);
  x->Init();
  x->tensor<float, 2>().setConstant(0.2);
  //LOG(INFO)  << "content of x:\n" << x->tensor<float, 2>();

  auto matrix_x = ToEigenMatrix<float>(*x);
  //LOG(INFO) << "matrix_x:\n" << matrix_x << "\nrows:" << matrix_x.rows() << ", cols:" << matrix_x.cols();

  Tensor* w = GetTensor(exec, "w");
  w->tensor<float, 2>().setConstant(0.3);
  //LOG(INFO)  << "content of w:\n" << w->tensor<float, 2>();

  auto matrix_w = ToEigenMatrix<float>(*w);
  //LOG(INFO) << "matrix_w:\n" << matrix_w << "\nrows:" << matrix_w.rows() << ", cols:" << matrix_w.cols();

  Tensor* b = GetTensor(exec, "b");
  LOG(INFO) << "b: " << b->shape().DebugString();
  b->vec<float>().setConstant(0.002);
  //LOG(INFO) << "content of b:\n" << b->vec<float>();
  
  // y = 0.139364
  Status s = exec.Run();
  
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
