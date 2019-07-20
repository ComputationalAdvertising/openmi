#include "tensor.h"
#include <assert.h>

namespace openmi {

Tensor::Tensor(): Tensor(DT_FLOAT) {}

Tensor::Tensor(DataType type): type_(type) {}

Tensor::Tensor(DataType type, const TensorShape& shape) 
  : shape_(shape), type_(type), alloc_(nullptr) {
  CHECK(shape.NumElements() > 0) 
    << "element number of tensor must be great than 0. "
    << "tensor shape: " << shape.DebugString();
  Init();
}

Tensor::Tensor(Allocator* alloc, DataType type, const TensorShape& shape)
  : shape_(shape), type_(type), alloc_(alloc) {
  CHECK(alloc != nullptr) << "alloc is nullptr when new Tensor.";
  CHECK(shape.NumElements() > 0) 
    << "element number of tensor must be great than 0. "
    << "tensor shape: " << shape.DebugString();
  Init();
}

Tensor::Tensor(proto::TensorProto& tensor)
  : shape_(tensor.tensor_shape()), type_(tensor.dtype()) {
  Init();
  // TODO copy tensor content to this
}

Tensor::Tensor(proto::TensorShapeProto& shape_proto, DataType type) 
  : shape_(shape_proto), type_(DT_FLOAT) {
  Init();
}

void Tensor::Init() {
  if (alloc_ == nullptr) {
    alloc_.reset(cpu_allocator());
  }
  assert(alloc_ != nullptr);
  size_t size = shape().NumElements() * SizeOfType(type_);
  buf_.reset(new TensorBuffer(alloc_.get(), size));
  if (buf_->IsInitialized()) {
    is_initialized_ = true;
  }
}

Tensor::~Tensor() {
}

} // namespace openmi
