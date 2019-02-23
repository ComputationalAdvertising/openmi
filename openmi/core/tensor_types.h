#ifndef OPENMI_CORE_FRAMEWORK_TYPES_H_
#define OPENMI_CORE_FRAMEWORK_TYPES_H_ 

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace openmi {

// Eigen::Matrix 
template <typename T>
struct MTypes {
  // matrix of scalar type T
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
}; 

// Eigen::TensorMap 
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
struct TTypes {
  // Rank-<NDIMS> tensor of scalar type T
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned> Tensor;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned> ConstTensor; 

  // Scalar tensor which implemented as a rank-0 tensor of scalar type T 
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>, Eigen::Aligned> Scalar;
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>, Eigen::Aligned> ConstScalar;

  // Rank-1 tesnor of scalar type T 
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned> Vector;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned> ConstVector;

  // Rank-2 tesnor of scalar type T 
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType>, Eigen::Aligned> Matrix;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>, Eigen::Aligned> ConstMatrix; 
};

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_TYPES_H_ 
