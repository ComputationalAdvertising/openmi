#include "base/logging.h"
#include "types.h"
#include "tensor.h"

using namespace openmi;

void tensor_buffer_test();
void matmul_test();
void tensor_basic_test();

int main(int argc, char** argv) {
  //matmul_test();
  tensor_buffer_test();
  tensor_basic_test();
  return 0;
}

/*
template <typename T>
Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
ToEigenMatrix(Tensor& tensor) {
  auto matrix = tensor.matrix<T>();
  return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(matrix.data(), matrix.dimension(0), matrix.dimension(1));
}
*/

void tensor_buffer_test() {
  size_t size = 10;
  Allocator* alloc = openmi::cpu_allocator();
  if (alloc == nullptr) {
    LOG(INFO) << "alloc is null";
    return;
  }
  std::shared_ptr<TensorBuffer> buf(new TensorBuffer(alloc, size));
  LOG(INFO) << "buf.size:" << buf->size();
}

void matmul_test() {
  std::string shapes1 = "2,3";
  TensorShape shape(shapes1);
  typedef EnumToDataType<DT_FLOAT>::T T;
  Tensor* t1 = new Tensor(DT_FLOAT, shape);
  //auto Y = ToEigenMatrix<T>(*t1);
  //Y.setConstant(5);
  //LOG(INFO) << "Y:\n" << Y;
  auto tt = t1->tensor<T, 2>();
  LOG(INFO) << "tt:\n" << tt;
}

void tensor_basic_test() {
  std::string shapes = "2,3,4";
  TensorShape shape(shapes);
  //typedef double T;
  typedef EnumToDataType<DT_FLOAT>::T T;
  Tensor* tensor = new Tensor(DT_FLOAT, shape);
  auto tt = tensor->tensor<T, 3>();  
  //auto = Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor, Eigen::DenseIndex>>
  LOG(INFO) << "d0: " << tt.dimensions()[0];
  LOG(INFO) << "d1: " << tt.dimensions()[1];
  LOG(INFO) << "d2: " << tt.dimensions()[2];
  //LOG(INFO) << "d3: " << tt.dimensions()[3];
  LOG(INFO) << "num_elements: " << tensor->Shape().NumElements();

  // = 
  tt(0,0,0) = 1; 
  tt(0,0,1) = 2;

  LOG(INFO) << "tensor:\n" << tt;
  
  // Eigen::Tensor init method
  tt.setConstant(0.1);
  LOG(INFO) << "tensor [Constant]:\n" << tt;
  
  tt.setRandom();
  LOG(INFO) << "tensor [Random]:\n" << tt;

  tt.setZero();
  LOG(INFO) << "tensor [Zero]:\n" << tt;
}
