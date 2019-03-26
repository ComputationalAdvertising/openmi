#include "tensor.h"
#include "tensor_shape.h"
#include "cwise_ops_binary.h"
#include "base/logging.h"
#include "base/register.h"
#include "device_registry.h"

Tensor& CreateTensor(std::string& shape_, const int dims) {
  TensorShape shape(shape_);
  Tensor* t = new Tensor(DT_FLOAT, shape);
  return *t;
}

int main(int argc, char** argv) {
  const int NDIMS = 2;
  std::string shape_a_("4,3");
  Tensor& a = CreateTensor(shape_a_, NDIMS);
  a.tensor<float, 2>().setConstant(10);
  
  std::string shape_b_("1");
  Tensor& b = CreateTensor(shape_b_, NDIMS);
  b.tensor<float, 1>().setConstant(20);

  Tensor* out = new Tensor(DT_FLOAT);

  typedef float T;
  Device* device = openmi::Register<DeviceFactory>::Find("CPU")->func();
  CHECK(device != nullptr) << "device not eixst. 'CPU'";

  OpKernelContext::Params* params = new OpKernelContext::Params();
  params->device = device;
  OpKernelContext* context = new OpKernelContext(params);

  ::openmi::BinaryElementWiseOp<CpuDevice, AddFunctor<T>, T> op;
  op.Operate<NDIMS>(context, a, b, *out);

  return 0;
}
