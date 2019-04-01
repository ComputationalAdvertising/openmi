#ifndef OPENMI_CORE_OPS_NUMERIC_OP_H_
#define OPENMI_CORE_OPS_NUMERIC_OP_H_ 

#include "op_kernel.h"
#include "op_registry.h"

namespace openmi {

typedef Eigen::ThreadPoolDevice CpuDevice;

/*!
 * \brief Unary op. it requires at least one input
 */ 
template <typename T, typename CHILD> 
class UnaryOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) override {
    CHECK(context->inputs().size() > 0) 
      << context->name() << " not input node.";
    
    Tensor& in = context->input(0);
    Tensor& out = context->output();
    LOG(DEBUG) << "in.name: " << context->inputs().at(0) << ", out.name: " << context->name() << ", related_node: " << context->related_node_name();
    if (!out.IsInitialized()) {
      LOG(DEBUG) << "not initialized";
      TensorShape out_shape;
      auto& related_node = context->GetTensor(
        context->related_node_name());
      if (related_node.IsInitialized()) {
        out_shape = related_node.shape();
      } else {
        out_shape = in.shape();
      }
      out.AllocateTensor(out_shape);
    }

    size_t max_dims = std::max(in.shape().Dims(), out.shape().Dims());
    switch (max_dims) {
#define NDIM_CASE(NDIMS)  \
    case NDIMS: { \
      static_cast<CHILD*>(this)->template Operate<NDIMS>(context, in, out); \
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
        LOG(ERROR) << "opnemi only support handle up to Tensor::dims() up to 8. not " << in.shape().Dims();
        break;     
    }
  }
}; // class UnaryOp 


/*!
 * \brief General Binary Operator
 */
template <typename T, typename CHILD>
class BinaryOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) override {
    auto& in0 = context->input(0);
    auto& in1 = context->input(1);
    auto& out = context->output();

    auto in0_dims = in0.shape().Dims();
    auto in1_dims = in1.shape().Dims();
    auto max_dims = std::max(in0_dims, in1_dims);

    if (!out.IsInitialized()) {
      TensorShape out_shape;
      auto& related_node = context->GetTensor(
        context->related_node_name());
      if (related_node.IsInitialized()) {
        out_shape = related_node.shape();
      } else {
        for (auto i = 0; i < max_dims; ++i) {
          auto in0_ith_dim = 1;
          if (in0_dims >= i && in0.shape().DimSize(i) > 1) {
            in0_ith_dim = in0.shape().DimSize(i);
          }
          auto in1_ith_dim = 1;
          if (in1_dims >= i && in1.shape().DimSize(i) > 1) {
            in1_ith_dim = in1.shape().DimSize(i);
          }
          out_shape.AddDim(
            in0_ith_dim > in1_ith_dim ? in0_ith_dim : in1_ith_dim
          );
        }
      }
      out.AllocateTensor(out_shape);
    }

    switch (max_dims) {
#define NDIM_CASE(NDIMS) \
  case NDIMS: {  \
    static_cast<CHILD*>(this)->template Operate<NDIMS>(context, in0, in1, out);  \
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
        LOG(ERROR) << "opnemi only support handle up to Tensor::dims() up to 8. not " << max_dims;
        break;
    }
  }
}; // class BinaryOp

template <typename T, typename F, typename R = T>
struct BaseFunctor {
  typedef F func;
  typedef R out_type;
  typedef T in_type;

  static const bool use_bcast_optimization = true;
};

} // namespace openmi 

// for unary op
#define OPENMI_REGISTER_UNARY_OP_WITH_TYPE(name, CHILD, T) \
  OPENMI_REGISTER_OP_KERNEL(name,  \
    ::openmi::UnaryOp<T, ::openmi::CHILD<CpuDevice, T>>)  \
    .Device("CPU") \
    .TypeConstraint<T>();

#define OPENMI_REGISTER_UNARY_OP(name, CHILD) \
  OPENMI_REGISTER_UNARY_OP_WITH_TYPE(name, CHILD, float) \
  OPENMI_REGISTER_UNARY_OP_WITH_TYPE(name, CHILD, double) \
  OPENMI_REGISTER_UNARY_OP_WITH_TYPE(name, CHILD, int)  

// for binary op 
#define OPENMI_REGISTER_BINARY_OP_WITH_TYPE(name, CHILD, T) \
  OPENMI_REGISTER_OP_KERNEL(name, \
      ::openmi::BinaryOp<T, CHILD<CpuDevice, T>>) \
    .Device("CPU") \
    .TypeConstraint<T>();

#define OPENMI_REGISTER_BINARY_OP(name, CHILD) \
  OPENMI_REGISTER_BINARY_OP_WITH_TYPE(name, CHILD, float) \
  OPENMI_REGISTER_BINARY_OP_WITH_TYPE(name, CHILD, double) \
  OPENMI_REGISTER_BINARY_OP_WITH_TYPE(name, CHILD, int) \

#endif // OPENMI_CORE_OPS_NUMERIC_OP_H_
