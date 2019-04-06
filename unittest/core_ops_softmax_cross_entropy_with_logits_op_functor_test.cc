#include "tensor.h"
#include "tensor_shape.h"
#include "base/logging.h"
#include "base/register.h"
#include "softmax_op_functor.h"
#include "cross_entropy_op_functor.h"
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
  Tensor& logits = CreateTensor(logit_shape, NDIMS);
  logits.tensor<float, 2>().setConstant(6);
  logits.tensor<float, 2>()(0,0) = 0.4;
  logits.tensor<float, 2>()(1,0) = 0.4;
  logits.tensor<float, 2>()(2,0) = 0.4;
  logits.tensor<float, 2>()(3,0) = 0.4;

  LOG(DEBUG) << "a:\n" << logits.tensor<float, 2>();
  
  Device* device = openmi::Register<DeviceFactory>::Find("CPU")->func();
  CHECK(device != nullptr) << "device not eixst. 'CPU'";

  std::string label_shape("4,2");
  Tensor& label = CreateTensor(label_shape, NDIMS);
  label.tensor<float, 2>().setConstant(1);
  label.tensor<float, 2>()(0,0) = 0;
  label.tensor<float, 2>()(1,0) = 0;
  label.tensor<float, 2>()(2,0) = 0;
  label.tensor<float, 2>()(3,0) = 0;

  LOG(DEBUG) << "label:\n" << label.tensor<float, 2>();

  std::string loss_shape("4,1");
  Tensor& loss = CreateTensor(loss_shape, NDIMS);
  loss.tensor<float, 2>().setConstant(0);

  OpKernelContext::Params* params = new OpKernelContext::Params();
  params->device = device;
  OpKernelContext* context = new OpKernelContext(params);
  
  typedef float T;
  
  ::openmi::functor::SoftmaxCrossEntropyWithLogitsImpl<CpuDevice, T>::Compute(
    context->template eigen_device<CpuDevice>(), label.matrix<float>(), logits.matrix<float>(), loss.matrix<float>());
  LOG(DEBUG) << "after softmax cross entropy with logits. loss:\n" << loss.tensor<float, 2>();
  return 0;
}
