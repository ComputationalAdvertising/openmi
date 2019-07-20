#include "matmul_op.h"

#include "local_device.h"
#include "op_registry.h"

using namespace openmi;

namespace openmi {

/*!
 * \brief matrix multpily 
 *    In0: m*n, In1:n*k -> Y: m*k
 */
template <typename Device, typename T>
class MatMulOp : public OpKernel {
public:
  void Initialize(OpKernelConstruction* context) override;

  void Compute(OpKernelContext* context) override;

private:
  bool transpose_a_ = false;
  bool transpose_b_ = false;
  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair_;
}; // class MatMul

template <typename Device, typename T>
void MatMulOp<Device, T>::Initialize(OpKernelConstruction* ctx) {
  ctx->GetAttr<bool>("transpose_a", &transpose_a_);
  ctx->GetAttr<bool>("transpose_b", &transpose_b_);
  dim_pair_[0].first = transpose_a_ ? 0 : 1;
  dim_pair_[0].second = transpose_b_ ? 1 : 0;
  DLOG(INFO) << __FUNCTION__ << " test_test, transpose_a:" << transpose_a_ << ", transpose_b:" << transpose_b_;
}

template <typename Device, typename T>
void MatMulOp<Device, T>::Compute(OpKernelContext* ctx) {
  auto& in0 = ctx->input(0);
  CHECK(in0.CheckTensorInitialized(ctx->inputs().at(0)));
  auto& in1 = ctx->input(1);
  CHECK(in1.CheckTensorInitialized(ctx->inputs().at(1)));

  auto& out = ctx->output();

  auto X0 = in0.matrix<T>();
  auto X1 = in1.matrix<T>();

  DLOG(INFO) << "in0: " << ctx->inputs().at(0) << ", transpose:" << transpose_a_ 
    << ", shape:" << in0.shape().DebugString() << ", value:\n" << X0;
  DLOG(INFO) << "in1: " << ctx->inputs().at(1) << ", transpose:" << transpose_b_ 
    << ", shape:" << in1.shape().DebugString() << ", value:\n" << X1;

  TensorShape expected_out_shape;
  int a_dim_remaining = 1 - dim_pair_[0].first;
  int b_dim_remaining = 1 - dim_pair_[0].second;
  expected_out_shape.AddDim(in0.shape().DimSize(a_dim_remaining));
  expected_out_shape.AddDim(in1.shape().DimSize(b_dim_remaining));

  if (!out.IsInitialized() || out.shape() != expected_out_shape) {
    out.AllocateTensor(expected_out_shape);

    auto* related_node = ctx->GetTensor(ctx->related_node_name());
    if (related_node != nullptr && related_node->IsInitialized() && related_node->shape() != expected_out_shape) {
      related_node->AllocateTensor(expected_out_shape);
    }
  }

  auto d = ctx->template eigen_device<Device>();
  auto Y = out.matrix<T>();

  Y.device(d) = X0.contract(X1, dim_pair_);
  /*
  MatMulImpl<Device, typename TTypes<T>::Matrix, 
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> >(d, Y, X0, X1, dim_pair_);
  */
  DLOG(DEBUG) << "name:" << ctx->name() << ", Y:\n" << Y;
}

#define REGISTER_MATMUL_OP(T) \
  OPENMI_REGISTER_OP_KERNEL(MatMul, MatMulOp<CpuDevice, T>)  \
    .Device("CPU") \
    .TypeConstraint<T>();

REGISTER_MATMUL_OP(float)
REGISTER_MATMUL_OP(double)
REGISTER_MATMUL_OP(int)

OPENMI_REGISTER_FILE_TAG(matmul_op);

}
