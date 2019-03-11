#ifndef OPENMI_CORE_TENSOR_SHAPE_H_
#define OPENMI_CORE_TENSOR_SHAPE_H_ 

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "tensor_types.h"
#include "base/logging.h"
#include "openmi/idl/proto/tensor_shape.pb.h"

namespace openmi {

class TensorShape {
public:
  // "4,5,6,7"
  TensorShape(std::string& shapes);
  TensorShape(const char* shapes);
  TensorShape(); 

  ~TensorShape();

  TensorShape(std::vector<uint64_t>& dims);

  TensorShape(const TensorShape& other);

  TensorShape(const proto::TensorShapeProto& proto);
  
  bool IsSameSize(const TensorShape& other) const;
  bool operator==(const TensorShape& other) const { return IsSameSize(other); }
  bool operator!=(const TensorShape& other) const { return !IsSameSize(other); }

  template <int NDIMS>
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> AsEigenDSizes() const;
  
  template <int NDIMS>
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> AsEigenDSizesWithPadding() const;
  
  void AddDim(uint64_t size);

  void SetDim(int d, uint64_t size);

  std::vector<uint64_t>& Shape() { return dims_; }
  
  // Return the number of dimensions in the tensor
  inline size_t Dims() const { return dims_.size(); }

  // Return the number of elements in dimensions `d`. Eigen::Tensor::dimensions()
  inline uint64_t DimSize(int d) const { return dims_[d]; } 

  inline uint64_t NumElements() const { 
    return num_elements_; 
  }

  // For debug messages. 
  std::string DebugString() const;
  //static std::string DebugString(const TensorShapeProto& proto);

private:
  void Init(std::string& shapes);

private:
  std::vector<uint64_t> dims_; 
  uint64_t num_elements_ = 1L;
}; // class TensorShape 

typedef std::shared_ptr<TensorShape> TensorShapePtr; 

template <int NDIMS>
Eigen::DSizes<Eigen::DenseIndex, NDIMS> TensorShape::AsEigenDSizes() const {
  return AsEigenDSizesWithPadding<NDIMS>();
}
  
template <int NDIMS>
Eigen::DSizes<Eigen::DenseIndex, NDIMS> TensorShape::AsEigenDSizesWithPadding() const {
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> dsizes;
  for (auto d = 0; d < Dims(); ++d) {
    dsizes[d] = DimSize(d);
  }
  
  for (auto d = Dims(); d < NDIMS; ++d) {
    dsizes[d] = 1;
  }
  return dsizes;
}

} // namespace openmi
#endif // OPENMI_CORE_TENSOR_SHAPE_H_ 
