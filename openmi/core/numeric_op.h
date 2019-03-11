#ifndef OPENMI_CORE_OPS_NUMERIC_OP_H_
#define OPENMI_CORE_OPS_NUMERIC_OP_H_ 

#include "op_kernel.h"

namespace openmi {

typedef Eigen::ThreadPoolDevice CpuDevice;

template <typename T>
class UnaryOp : public OpKernel {
public:
  virtual void Initialize(OpKernelConstruction* context) {
    // TODO 
  }
}; // class UnaryOp

// input.shape = output.shape
template <typename T, typename CHILD> 
class UnaryElementWiseOp : public UnaryOp<T> {
public:
  using UnaryOp<T>::UnaryOp;

  void Compute(OpKernelContext* context) override {
    Tensor& input = context->input(0);
    Tensor& output = context->output();
    if (!output.IsInitialized()) {
      output.AllocateTensor(input.shape());
    }

    if (input.shape() != output.shape()) {
      // TODO tensor.clear and reallocate memory  
      output.AllocateTensor(input.shape());
    }

    // TODO 重写判断条件
    static_cast<CHILD*>(this)->Operate(context, input, output);
  }

}; // class UnaryElementWiseOp

template <typename T> 
class BinaryOp : public OpKernel {
public:
  virtual void Initialize(OpKernelConstruction* context) {
    // TODO
  }

protected:
  bool reshape_ = true;
}; // class BinaryOp


template <typename T, typename CHILD>
class BinaryElementWiseOp : public BinaryOp<T> {
public:
  using BinaryOp<T>::BinaryOp;

  void Compute(OpKernelContext* context) override {
    auto& a = context->input(0);
    auto& b = context->input(1);
    auto& output = context->output();

    /*
    if (!reshape_) {
      // TODO check whether input shape is same  
    }
    */
    
    if (!output.IsInitialized()) {
      output.set_shape(a.shape());
      output.Init();
    }

    switch (a.shape().Dims()) {
#define NDIM_CASE(NDIMS) \
  case NDIMS: {  \
    static_cast<CHILD*>(this)->template Operate<NDIMS>(context, a, b, output);  \
    break; \
  }
      NDIM_CASE(1);
      NDIM_CASE(2);
      NDIM_CASE(3);
      NDIM_CASE(4);
      NDIM_CASE(5);
      NDIM_CASE(6);
      NDIM_CASE(7);
      NDIM_CASE(8);
#undef NDIM_CASE 
      
      default:
        //std::runtime_error("opnemi only support handle up to Tensor::dims() up to 8. not ", a.shape().Dims());
        LOG(ERROR) << "opnemi only support handle up to Tensor::dims() up to 8. not " << a.shape().Dims();
        break;
    }
  }

}; // class BinaryElementWiseOp

} // namespace openmi 

#define OPENMI_REGISTER_UNARY_OP(name, CHILD, T) \
  OPENMI_REGISTER_OP_KERNEL(name, CHILD<CpuDevice, T>) \
    .Device("CPU")  \
    .TypeConstraint<T>();

#define OPENMI_REGISTER_UNARY_ELEMENT_WISE_OP_WITH_TYPE(name, CHILD, T) \
  OPENMI_REGISTER_OP_KERNEL(name,  \
      ::openmi::UnaryElementWiseOp<T, CHILD<CpuDevice, T>>)  \
    .Device("CPU") \
    .TypeConstraint<T>();

#define OPENMI_REGISTER_UNARY_ELEMENT_WISE_OP(name, CHILD) \
  OPENMI_REGISTER_UNARY_ELEMENT_WISE_OP_WITH_TYPE(name, CHILD, float) \
  OPENMI_REGISTER_UNARY_ELEMENT_WISE_OP_WITH_TYPE(name, CHILD, double) \
  OPENMI_REGISTER_UNARY_ELEMENT_WISE_OP_WITH_TYPE(name, CHILD, int)  

// for binary op 

#define OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP_WITH_TYPE(name, CHILD, T) \
  OPENMI_REGISTER_OP_KERNEL(name, \
      ::openmi::BinaryElementWiseOp<T, CHILD<CpuDevice, T>>) \
    .Device("CPU") \
    .TypeConstraint<T>();

#define OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP(name, CHILD) \
  OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP_WITH_TYPE(name, CHILD, float) \
  OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP_WITH_TYPE(name, CHILD, double) \
  OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP_WITH_TYPE(name, CHILD, int) \

#endif // OPENMI_CORE_OPS_NUMERIC_OP_H_
