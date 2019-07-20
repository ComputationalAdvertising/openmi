#ifndef OPENMI_CORE_TENSOR_H_
#define OPENMI_CORE_TENSOR_H_ 

#include "tensor_shape.h"
#include "tensor_types.h"
#include "types.h"
#include "base/allocator.h"
#include "base/logging.h"
#include "base/refcount.h"
#include "openmi/idl/proto/tensor.pb.h"
#include "openmi/idl/proto/tensor_shape.pb.h"

using namespace openmi;

namespace openmi { 

class TensorBuffer : public base::RefCounted {
public:
  TensorBuffer(Allocator* alloc, size_t size) 
  : alloc_(alloc), size_(size), is_initialized_(false) {
    if (alloc_ == nullptr) {
      LOG(ERROR) << "alloc_ is null. return";
      return;
    }
    data_ = alloc_->AllocateRaw(size_);
    if (data_ != nullptr) {
      is_initialized_ = true;
    }
  }

  void* data() { return data_; }
  size_t size() const { return size_; }

  bool IsInitialized() const { return is_initialized_; }

  template <typename T>
  T* Base() {
    return reinterpret_cast<T*>(data());
  }

  void Clear() {
    if (data_ != nullptr) {
      alloc_->DeallocateRaw(data_);
      data_ = nullptr;
      is_initialized_ = false;
    }
  }

  bool OwnsMemory() const { return true; }

private:
  Allocator* alloc_;
  void* data_;
  size_t size_;
  bool is_initialized_;
}; // class TensorBuffer 

class Tensor {
public:
  Tensor();

  explicit Tensor(DataType type);

  Tensor(DataType type, const TensorShape& shape);
  
  Tensor(Allocator* alloc, DataType type, const TensorShape& shape);

  Tensor(proto::TensorProto& tensor);

  Tensor(proto::TensorShapeProto& shape_proto, DataType type = DT_FLOAT);

  ~Tensor();

  bool IsInitialized() const { 
    return is_initialized_; 
  }

  const DataType type() const { 
    return type_; 
  }

  bool IsVector() const {
    return shape_.Dims() == 1 && shape_.NumElements() == shape_.DimSize(0);
  }

  void set_shape(TensorShape& shape) {
    shape_ = shape;
  } 

  TensorShape& shape() { return shape_; }

  template <typename T>
  T* Base() { return buf_->Base<T>(); }

  template <typename T, size_t NDIMS> 
  typename TTypes<T, NDIMS>::Tensor tensor();

  template <typename T>
  typename TTypes<T>::Vector vec() {
    return tensor<T, 1>();
  }

  template <typename T>
  typename TTypes<T>::Matrix matrix() {
    return tensor<T, 2>();
  }

  template <typename T>
  typename TTypes<T>::ConstMatrix matrix() const {
    return tensor<T, 2>();
  }

  template <typename T>
  typename TTypes<T>::Flat flat() {
    std::vector<uint64_t> new_sizes;
    new_sizes.push_back(shape().NumElements());
    return shaped<T, 1>(new_sizes);
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor shaped(
    std::vector<uint64_t> new_sizes) {
    Eigen::array<Eigen::DenseIndex, NDIMS> dims;
    for (auto i = 0; i < new_sizes.size(); ++i) {
      dims[i] = new_sizes[i];
    }
    return typename TTypes<T, NDIMS>::Tensor(Base<T>(), dims);
  }

  template <typename T>
  typename MTypes<T>::MatrixRef ToEigenMatrix() {
    auto m = matrix<T>();
    return Eigen::Map<typename MTypes<T>::Matrix>(
      m.data(), m.dimension(0), m.dimension(1));
  }

  // RowMajor
  template <typename T>
  typename MTypes<T>::MatrixRef ToEigenVector() {
    auto v = vec<T>();
    return Eigen::Map<typename MTypes<T>::Matrix>(
      v.data(), 1, v.dimension(0));
  }

//private:
  void Init(); 

  void AllocateTensor(TensorShape& shape) {
    set_shape(shape);
    if (alloc_ == nullptr) {
      DLOG(INFO) << "alloc_ is null. reset it.";
      alloc_.reset(cpu_allocator());
    }
    size_t size = shape_.NumElements() * SizeOfType(type_);
    if (buf_ != nullptr) {
      buf_->Clear();
    }
    buf_.reset(new TensorBuffer(alloc_.get(), size));
    if (buf_->IsInitialized()) {
      is_initialized_ = true;
    } else {
      LOG(FATAL) << "tensor buffer initialize failed.";
    }
  }

  void ReallocateTensor(TensorShape& shape) {
    if (buf_ != nullptr) {
      buf_->Clear();
    }
    AllocateTensor(shape);
  }

  inline bool CheckTensorInitialized(std::string& name) {
    if (!IsInitialized()) {
      LOG(ERROR) << ". input tensor has not initialized. "
        << " please check tensor init or allocate buffer. "
        << "name[" << name << "]";
      return false;
    }
    return true;
  }

private:
  TensorShape shape_;
  DataType type_;
  std::shared_ptr<Allocator> alloc_;
  std::shared_ptr<TensorBuffer> buf_;
  bool is_initialized_ = false;
}; // class Tensor

template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor Tensor::tensor() {
  CHECK(is_initialized_ == true) << "tensor has been not initialized.";
  return typename TTypes<T, NDIMS>::Tensor(
    Base<T>(), shape_.AsEigenDSizes<NDIMS>());
}

} // namespace openmi
#endif // OPENMI_CORE_TENSOR_H_
