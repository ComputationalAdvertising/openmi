#include "tensor_shape.h"
#include "base/dstring.h"
#include "base/type_conversion.h"

namespace openmi {

TensorShape::TensorShape() {
}

TensorShape::TensorShape(const char* _shapes) {
  std::string shapes(_shapes);
  Init(shapes);
}

TensorShape::TensorShape(std::string& shapes) {
  Init(shapes);
}

TensorShape::TensorShape(std::vector<uint64_t>& dims) : dims_(dims) {
  for (size_t i = 0; i < dims_.size(); ++i) {
    num_elements_ *= dims_[i];
  }
}

TensorShape::TensorShape(const TensorShape& other) 
  : dims_(other.dims_), num_elements_(other.num_elements_) {
}

TensorShape::TensorShape(const proto::TensorShapeProto& shape_proto) {
  for (size_t i = 0; i < shape_proto.dim().size(); ++i) {
    dims_.emplace_back(shape_proto.dim(i).size());
    num_elements_ *= shape_proto.dim(i).size();
  }
  LOG(DEBUG) << "TensorShape(TensorShapeProto) num_elements_:" << num_elements_;
}

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

void TensorShape::SetDim(int d, uint64_t size) {
  CHECK(d < dims_.size()) 
    << d << " extends out of range. dims_:" << dims_.size();
  num_elements_ /= dims_[d];
  dims_[d] = size;
  num_elements_ *= size;
}

void TensorShape::DeleteDim(int index) {
  CHECK(index < dims_.size()) 
    << index << " extends out of range. dims_:" << dims_.size();
  num_elements_ /= dims_[index];
  dims_.erase(dims_.begin() + index, dims_.begin() + index + 1);
}

bool TensorShape::IsSameSize(const TensorShape& other) const {
  if (dims_.size() != other.dims_.size()) {
    return false;
  }
  for (int i = 0; i < dims_.size(); ++i) {
    if (DimSize(i) != other.DimSize(i)) {
      return false;
    }
  }
  return true;
}

std::string TensorShape::DebugString() const {
  std::string debug ("num_elements_:" + std::to_string(num_elements_) + ", dims: " + std::to_string(Dims()));
  debug += ", shapes: ";
  for (int i = 0; i < Dims(); ++i) {
    debug += std::to_string(DimSize(i)) + ", ";
  }
  return debug;
}

}
