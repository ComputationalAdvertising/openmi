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
  LOG(INFO) << "TensorShape std::string shapes";
  Init(shapes);
}

TensorShape::TensorShape(const TensorShape& other): dims_(other.dims_), num_elements_(other.num_elements_) {
  LOG(INFO) << "copy TensorShape& other";
}

TensorShape::TensorShape(const pb::TensorShapeProto& shape_proto) {
  for (size_t i = 0; i < shape_proto.shape().size(); ++i) {
    dims_.emplace_back(shape_proto.shape(i));
    num_elements_ *= shape_proto.shape(i);
  }
  LOG(DEBUG) << "TensorShape(TensorShapeProto) num_elements_:" << num_elements_;
}

TensorShape::~TensorShape() {
}

void TensorShape::Init(std::string& shapes) {
  std::vector<std::string> dim_tokens;
  openmi::Split(shapes, &dim_tokens, ',');
  for (size_t i = 0; i < dim_tokens.size(); ++i) {
    size_t dim_size = openmi::StringToNum<size_t>(dim_tokens[i]);
    dims_.emplace_back(dim_size);
    num_elements_ *= dim_size;
  }
}

void TensorShape::AddDim(size_t size) {
  dims_.emplace_back(size);
  num_elements_ *= size;
}

void TensorShape::InsertDim(int d, size_t size) {
  dims_.insert(dims_.begin() + d, size);
  num_elements_ *= size;
}

bool TensorShape::IsSameSize(const TensorShape& other) const {
  // TODO 
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
