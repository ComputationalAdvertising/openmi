#include "op_kernel.h"

namespace openmi {

OpKernel::OpKernel(): context_(nullptr), initialized_(false) {}

OpKernel::~OpKernel() {}

void OpKernel::Initialize(OpKernelConstruction* context) {
  context_ = context;
  initialized_ = true;
}

OpKernelContext::OpKernelContext(Params* params): params_(params) {}

OpKernelContext::~OpKernelContext() {}

/*
Status OpKernelContext::Allocate(TensorShape& shape, DataType type) {
  // TODO 
  return Status::Ok();
}
  */

}
