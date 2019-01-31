#include "tensor_shape.h"
#include "base/dstring.h"
#include "base/type_conversion.h"

namespace openmi {

TensorShape::TensorShape(size_t dims) {
  dims_.resize(dims, 1L);
  num_elements_ = dims * 1;
}

TensorShape::TensorShape(const char* _shapes) {
  LOG(INFO) << "_shapes";
  std::string shapes(_shapes);
  Init(shapes);
}

TensorShape::TensorShape(std::string& shapes) {
  LOG(INFO) << "shapes";
  Init(shapes);
}
/*
TensorShape::TensorShape(const TensorShape& other): dims_(other.dims_), num_elements_(other.num_elements_) {
  LOG(INFO) << "TensorShape";
}
*/

TensorShape::~TensorShape() {
}

void TensorShape::Init(std::string& shapes) {
  std::vector<std::string> dim_tokens;
  openmi::Split(shapes, &dim_tokens, ',');
  for (size_t i = 0; i < dim_tokens.size(); ++i) {
    uint64_t dim_size = openmi::StringToNum<uint64_t>(dim_tokens[i]);
    dims_.emplace_back(dim_size);
    num_elements_ *= dim_size;
  }
}

void TensorShape::AddDim(uint64_t size) {
  dims_.emplace_back(size);
  num_elements_ *= size;
}

void TensorShape::InsertDim(int d, uint64_t size) {
  dims_.insert(dims_.begin() + d, size);
  num_elements_ *= size;
}

bool TensorShape::IsSameSize(const TensorShape& other) const {
  // TODO 
  return true;
}

std::string TensorShape::DebugString() const {
  std::string debug ("num_elements_:" + std::to_string(num_elements_));
  return debug;
}

}
