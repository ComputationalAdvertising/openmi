#include "tensor.h"
#include "tensor_shape.h"
#include "base/logging.h"
#include "base/register.h"
#include "softmax_op_functor.h"
#include "softmax_cross_entropy_with_logits_op_functor.h"
#include "op_kernel.h"
#include "device_registry.h"

Tensor& CreateTensor(std::string& shape_, const int dims) {
  TensorShape shape(shape_);
  Tensor* t = new Tensor(DT_FLOAT, shape);
  return *t;
}

int main(int argc, char** argv) {
  const int NDIMS = 2;
  std::string logit_shape("4,2");
  Tensor& a = CreateTensor(logit_shape, NDIMS);
  a.tensor<float, 2>().setConstant(6);
  a.tensor<float, 2>()(0,0) = 0.4;
  a.tensor<float, 2>()(1,0) = 0.4;
  a.tensor<float, 2>()(2,0) = 0.4;
  a.tensor<float, 2>()(3,0) = 0.4;

  LOG(DEBUG) << "a:\n" << a.tensor<float, 2>();
  
  Tensor& b = CreateTensor(logit_shape, NDIMS);
  b.tensor<float, 2>().setConstant(0);
  
  typedef float T;
  Device* device = openmi::Register<DeviceFactory>::Find("CPU")->func();
  CHECK(device != nullptr) << "device not eixst. 'CPU'";
  OpKernelContext::Params* params = new OpKernelContext::Params();
  params->device = device;
  OpKernelContext* context = new OpKernelContext(params);
  
  ::openmi::functor::SoftmaxImpl<CpuDevice, T>::Compute(context->template eigen_device<CpuDevice>(), a.matrix<float>(), b.matrix<float>(), false);
  LOG(DEBUG) << "after softmax. b:\n" << b.tensor<float, 2>();
  
  ::openmi::functor::SoftmaxImpl<CpuDevice, T>::Compute(context->template eigen_device<CpuDevice>(), a.matrix<float>(), b.matrix<float>(), true);
  LOG(DEBUG) << "after softmax. b:\n" << b.tensor<float, 2>();

  return 0;
}
