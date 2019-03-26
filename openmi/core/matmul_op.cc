#include "matmul_op.h"

#include "local_device.h"
#include "op_registry.h"

using namespace openmi;

namespace openmi {

DataType tmp(DataType type) {
  switch(type) {
    case DT_FLOAT:
      return DT_FLOAT;
    case DT_DOUBLE:
      return DT_DOUBLE;
    default:
      LOG(ERROR) << "Unrecognized DataType. " << type;
      return DT_INVALID;
  }
}

void MatMul::Initialize(OpKernelConstruction* ctx) {
  auto it = ctx->attrs().find("transpose_a");
  if (it != ctx->attrs().end()) {
    CHECK(it->second.attr_type == ::openmi::AttrValue::kBool);
    transpose_a_ = it->second.b;
  }

  it = ctx->attrs().find("transpose_b");
  if (it != ctx->attrs().end()) {
    CHECK(it->second.attr_type == ::openmi::AttrValue::kBool);
    transpose_b_ = it->second.b;
  }
  
  dim_pair_[0].first = transpose_a_ ? 0 : 1;
  dim_pair_[0].second = transpose_b_ ? 1 : 0;

  LOG(INFO) << "MatMul::Initialize done";
}

void MatMul::Compute(OpKernelContext* ctx) {
  Tensor* a = nullptr;
  Status status = ctx->session_state()->GetTensor(ctx->inputs()[0], &a);
  Tensor* b = nullptr;
  status = ctx->session_state()->GetTensor(ctx->inputs()[1], &b);

  LOG(INFO) << "a: " << a->shape().DebugString();
  LOG(INFO) << "b: " << b->shape().DebugString();
  
  Tensor* out = nullptr;
  std::string out_name = ctx->name();
  ctx->session_state()->GetTensor(out_name, &out);
  if (!out->IsInitialized()) {
    TensorShape out_shape;
    int a_dim_remaining = 1 - dim_pair_[0].first;
    int b_dim_remaining = 1 - dim_pair_[0].second;
    out_shape.AddDim(a->shape().DimSize(a_dim_remaining));
    out_shape.AddDim(b->shape().DimSize(b_dim_remaining));
    out->set_shape(out_shape);
    out->Init();
    LOG(INFO) << "xw^T shape: " << out_shape.DebugString();
  }

  Allocator* allocator = cpu_allocator();
  LocalDevice device(2);
  device.set_allocator(allocator);
  const Eigen::ThreadPoolDevice& d = device.eigen_cpu_device();

  auto A = a->tensor<float, 2>();
  auto B = b->tensor<float, 2>();
  auto Y = out->tensor<float, 2>();

  MatMulImpl<CpuDevice, TTypes<float, 2>::Tensor, 
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> >(d, Y, A, B, dim_pair_);

  //Y.device(d) = A.contract(B, dim_pair_);
  
  LOG(INFO) << "X1:\n" << A << ", transpose_a:" << transpose_a_;
  LOG(INFO) << "X2:\n" << B << ", transpose_b:" << transpose_b_; 
  LOG(INFO) << "name:" << out_name << ", Y:\n" << Y << "\nsizeof(tensor::type): " << SizeOfType(out->type());
}

/**
void MatMul::Compute(OpKernelContext* ctx) {
  Tensor* a = nullptr;
  Status status = ctx->session_state()->GetTensor(ctx->inputs()[0], &a);
  Tensor* b = nullptr;
  status = ctx->session_state()->GetTensor(ctx->inputs()[1], &b);

  auto it = ctx->attrs().find("transpose_a");
  if (it != ctx->attrs().end()) {
    CHECK(it->second.attr_type == ::openmi::AttrValue::kBool);
    transpose_a_ = it->second.b;
  }

  it = ctx->attrs().find("transpose_b");
  if (it != ctx->attrs().end()) {
    CHECK(it->second.attr_type == ::openmi::AttrValue::kBool);
    transpose_b_ = it->second.b;
  }

  TensorShape shape;
  if (transpose_a_) {
    shape.AddDim(a->shape().DimSize(1));
  } else {
    shape.AddDim(a->shape().DimSize(0));
  }

  if (transpose_b_) {
    shape.AddDim(b->shape().DimSize(0));
  } else {
    shape.AddDim(b->shape().DimSize(1));
  }

  LOG(INFO) << "xw^T shape: " << shape.DebugString();

  Tensor* y = nullptr;
  std::string out_name = ctx->name();
  status = ctx->session_state()->GetTensor(out_name, &y);
  if (!y->IsInitialized()) {
    y->set_shape(shape);
    y->Init();
    // TODO initialize tensor
  }

  //typedef EnumToDataType<a->type()>::T TA;
  auto X1 = a->ToEigenMatrix<float>();
  //typedef EnumToDataType<b->type()>::T TB;
  auto X2 = b->ToEigenMatrix<float>();
  //typedef EnumToDataType<y->type()>::T TY;
  auto Y = y->ToEigenMatrix<float>();

  if (!transpose_a_ && !transpose_b_) {
    Y.noalias() = X1 * X2;
  } else if (!transpose_a_ && transpose_b_) {
    Y.noalias() = X1 * X2.transpose();
  } else if (transpose_a_ && !transpose_b_) {
    Y.noalias() = X1.transpose() * X2;
  } else {
    Y.noalias() = X1.transpose() * X2.transpose();
  }

  LOG(INFO) << "X1:\n" << X1 << ", transpose_a:" << transpose_a_;
  LOG(INFO) << "X2:\n" << X2 << ", transpose_b:" << transpose_b_; 
  LOG(INFO) << "name:" << out_name << ", Y:\n" << Y << "\nsizeof(tensor::type): " << SizeOfType(y->type());

  //DataType dt = a->type();
  //const DataType newdt = tmp(dt);
  //typedef EnumToDataType<newdt>::T T;
  //LOG(INFO) << "sizeof: " << sizeof(T);
}
*/

//OPENMI_REGISTER_OP_KERNEL(MatMul, MatMul).Device("CPU");

#define REGISTER_MATMUL(T) \
  OPENMI_REGISTER_OP_KERNEL(MatMul, MatMul)  \
    .Device("CPU") \
    .TypeConstraint<T>();

REGISTER_MATMUL(float)
REGISTER_MATMUL(double)
REGISTER_MATMUL(int)

OPENMI_REGISTER_FILE_TAG(MatMul);

}
