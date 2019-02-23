#include "tensor_shape.h"
#include "base/logging.h"
using namespace openmi;

int main(int argc, char** argv) {
  std::string shapes("2,7,5,3");
  openmi::TensorShape tensor_shape(shapes);
  std::vector<uint64_t> dims = tensor_shape.Shape();
  for (size_t i = 0; i < dims.size(); ++i) {
    LOG(INFO) << "i: " << i << ", dim: " << dims[i];
  }
  LOG(INFO) << "num_element: " << tensor_shape.NumElements();
  return 0;
}
