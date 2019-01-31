#ifndef OPENMI_CORE_TENSOR_BUFFER_H_
#define OPENMI_CORE_TENSOR_BUFFER_H_ 

#include <memory>
#include <stddef.h>
#include "base/sarray.h"

namespace openmi {

/**
 * Notice: non thread-safe 
 */
template <typename T>
class TensorBuffer {
public:
  TensorBuffer(T* data, uint64_t size) {
    ptr_.reset(new openmi::SArray<T>(data, size));
  }

  TensorBuffer(uint64_t size) {
    ptr_.reset(new openmi::SArray<T>(size));
  }

  ~TensorBuffer() {
  }

  T* Data() const { return ptr_->data(); }

  uint64_t Size() const { return ptr_->size(); }

  uint64_t MemoryBytes() const {
    return ptr_->size() * sizeof(T);
  }
private:
  std::shared_ptr<openmi::SArray<T> > ptr_;
}; // class TensorBuffer

} // namespace openmi
#endif // OPENMI_CORE_TENSOR_BUFFER_H_
