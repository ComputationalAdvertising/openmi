#include "op_kernel.h"

namespace openmi {

OpKernel::OpKernel() {}

OpKernel::~OpKernel() {}

void OpKernel::Initialize(OpKernelConstruction* context) {
  // TODO used for check params before compute
}

OpKernelContext::OpKernelContext(Params* params): params_(params) {}

OpKernelContext::~OpKernelContext() {}

template <>
const Eigen::ThreadPoolDevice& OpKernelContext::eigen_device() const {
  CHECK(params_->device != nullptr) << "OpKernelContext device is null";
  return params_->device->eigen_cpu_device();
}

/*
Status OpKernelContext::Allocate(TensorShape& shape, DataType type) {
  // TODO 
  return Status::Ok();
}
  */

}
