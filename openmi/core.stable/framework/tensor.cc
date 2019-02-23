#include "tensor.h"
#include <assert.h>

namespace openmi {

Tensor::Tensor(): Tensor(DT_FLOAT) {}

Tensor::Tensor(DataType type): type_(type) {}

Tensor::Tensor(DataType type, const TensorShape& shape) 
  : type_(type), shape_(shape), alloc_(nullptr) {
  Init();
}

Tensor::Tensor(Allocator* alloc, DataType type, const TensorShape& shape)
  : type_(type), shape_(shape), alloc_(alloc) {
  Init();
}

Tensor::Tensor(pb::TensorShapeProto& shape_proto) 
  : type_(DT_FLOAT), shape_(shape_proto) {
  Init();
}

void Tensor::Init() {
  if (alloc_ == nullptr) {
    alloc_.reset(cpu_allocator());
  }
  assert(alloc_ != nullptr);
  size_t size = Shape().NumElements() * SizeOfType(type_);
  buf_.reset(new TensorBuffer(alloc_.get(), size));
  if (buf_->IsInitialized()) {
    is_initialized_ = true;
  }
}

Tensor::~Tensor() {
}

} // namespace openmi
