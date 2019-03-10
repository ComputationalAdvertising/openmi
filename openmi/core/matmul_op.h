#ifndef OPENMI_CORE_KERNELS_MATMUL_OP_H_
#define OPENMI_CORE_KERNELS_MATMUL_OP_H_ 

#include "op_kernel.h"

namespace openmi {

// y = x * w^T
class MatMul : public OpKernel {
public:
  void Initialize(OpKernelConstruction* context) override;

  void Compute(OpKernelContext* context) override;

private:
  bool transpose_a_ = false;
  bool transpose_b_ = false;
  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair_;
}; // class MatMul

} // namespace openmi
#endif // OPENMI_CORE_KERNELS_MATMUL_OP_H_
