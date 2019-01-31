#include "core/framework/tensor.h"
#include "base/logging.h"
#include "types.h"

using namespace openmi;

int main(int argc, char** argv) {
  std::string shapes = "2,3,4";
  //typedef double T;
  typedef EnumToDataType<DT_DOUBLE>::T T;
  Tensor<T>* tensor = new Tensor<T>(shapes);
  auto tt = tensor->TensorType<3>();  
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

  return 0;
}
