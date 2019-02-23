#include "core/framework/op_kernel.h"

namespace openmi {

OpKernelContext::OpKernelContext(Params* params): params_(params) {}

OpKernelContext::~OpKernelContext() {}

Status OpKernelContext::Allocate(TensorShape& shape, DataType type) {
  // TODO 
  return Status::Ok();
}

}
