#include "core/framework/tensor_shape.h"
#include <iostream>

int main(int argc, char** argv) {
  std::string shapes("2,7,5,3");
  openmi::TensorShape tensor_shape(shapes);
  std::vector<uint64_t> dims = tensor_shape.Shape();
  for (size_t i = 0; i < dims.size(); ++i) {
    std::cout << "i: " << i << ", dim: " << dims[i] << std::endl;
  }
  std::cout << "num_element: " << tensor_shape.NumElements() << std::endl;
  return 0;
}
