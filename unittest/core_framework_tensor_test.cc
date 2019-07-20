#include "base/logging.h"
#include "types.h"
#include "tensor.h"

using namespace openmi;

void tensor_buffer_test();
void matmul_test();
void tensor_basic_test();
void tensor_contract_test();

int main(int argc, char** argv) {
  //matmul_test();
  //tensor_buffer_test();
  tensor_basic_test();
  //tensor_contract_test();
  return 0;
}

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
  const int DIMS = t1->shape().Dims();
  constexpr int consti = 10;
  int i = 10;
  if (consti == i) {
    LOG(INFO) << "-------- equal";
  }
  auto tt = t1->tensor<T, 2>();
  tt.setConstant(2);
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
  LOG(INFO) << "num_elements: " << tensor->shape().NumElements();

  // = 
  tt(0,0,0) = 1; 
  tt(0,0,1) = -1;

  LOG(INFO) << "tensor:\n" << tt; 

  LOG(INFO) << "tensor->flat():\n" << tensor->flat<T>();

  auto ttt = tt.cwiseMax(static_cast<float>(0));
  LOG(INFO) << "----------- after Relu:\n" << ttt;
  
  // Eigen::Tensor init method
  tt.setConstant(0.1);
  LOG(INFO) << "tensor [Constant]:\n" << tt;
  
  tt.setRandom();
  LOG(INFO) << "tensor [Random]:\n" << tt;

  tt.setZero();
  LOG(INFO) << "tensor [Zero]:\n" << tt;
}

void tensor_contract_test() {
  Eigen::Tensor<int, 2> a(2, 3);
  a.setValues({{1,2,3}, {1,2,3}});
  Eigen::Tensor<int, 2> b(3, 2);
  b.setValues({{1,2}, {3,4}, {5,6}});

  LOG(INFO) << "a:\n" << a;
  LOG(INFO) << "b:\n" << b;
  
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {{Eigen::IndexPair<int>(1, 0)}};
  Eigen::Tensor<int, 2> ab = a.contract(b, product_dims);
  LOG(INFO) << "ab:\n" << ab;

  // Compute the product of the transpose of the matrices
  Eigen::array<Eigen::IndexPair<int>, 1> transposed_product_dims = {{Eigen::IndexPair<int>(0, 0)}};
  Eigen::Tensor<int, 2> AtBt = a.contract(b, transposed_product_dims);
  LOG(INFO) << "aTbT:\n" << AtBt;

  Eigen::array<Eigen::IndexPair<int>, 2> double_contraction_product_dims = {{Eigen::IndexPair<int>(0, 0), Eigen::IndexPair<int>(1, 1)}};
  Eigen::Tensor<int, 0> AdoubleContractedA = a.contract(a, double_contraction_product_dims);
  LOG(INFO) << "AdoubleContractedA:\n" << AdoubleContractedA;
  
}
