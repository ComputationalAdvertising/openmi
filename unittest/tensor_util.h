#include "tensor.h"
#include "tensor_shape.h"
using namespace openmi;

namespace openmi {
namespace test {

inline Tensor& CreateTensor(std::string& _shape, const int dims, DataType type) {
  TensorShape shape(_shape);
  Tensor* t = new Tensor(type, shape);
  return *t;
}

} // namespace test
} // namespace openmi
