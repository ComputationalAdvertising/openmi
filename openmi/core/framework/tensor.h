#ifndef OPENMI_CORE_TENSOR_H_
#define OPENMI_CORE_TENSOR_H_ 

#include "tensor_buffer.h"
#include "tensor_shape.h"
#include "tensor_types.h"
#include "base/logging.h"

namespace openmi {

template <typename T = float>
class Tensor {
public:
  Tensor();

  Tensor(std::string& shapes);

  Tensor(TensorShape& tensor_shape);

  ~Tensor();

  inline TensorShape& Shape() { 
    return tensor_shape_; 
  }

  template <size_t NDIMS> 
  typename TTypes<T, NDIMS>::Tensor TensorType();

  typename TTypes<T>::Vector Vec() {
    return TensorType<T, 1>();
  }

  typename TTypes<T>::Matrix MatrixType() {
    return TensorType<T, 2>();
  }

  //template <size_t NDIMS> 
  //typename TTypes<T, NDIMS>::Tensor Shaped();

private:
  void Init();

private:
  TensorShape tensor_shape_;
  std::shared_ptr<TensorBuffer<T> > buf_;
}; // class Tensor

template <typename T>
Tensor<T>::Tensor() {
  Init();
}

template <typename T>
Tensor<T>::Tensor(std::string& shapes): tensor_shape_(shapes) {
  Init();
}

template <typename T>
Tensor<T>::Tensor(TensorShape& tensor_shape): tensor_shape_(tensor_shape) {
  Init();
}

template <typename T> 
void Tensor<T>::Init() {
  uint64_t num_elements = Shape().NumElements();
  buf_.reset(new TensorBuffer<T>(num_elements));
}

template <typename T>
Tensor<T>::~Tensor() {
}

template <typename T>
template <size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor<T>::TensorType() {
  return typename TTypes<T, NDIMS>::Tensor(buf_->Data(), tensor_shape_.AsEigenDSizes<NDIMS>());
}

} // namespace openmi
#endif // OPENMI_CORE_TENSOR_H_
