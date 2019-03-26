#include "cwise_ops_binary.h"
#include "op_registry.h"

namespace openmi {

void UpdateOneVectorReshape(Tensor& t, uint64_t* reshape, int dim_size) {
  reshape[0] = 1;
  reshape[1] = t.shape().DimSize(0);
  for (size_t i = 2; i < dim_size; ++i) {
    reshape[i] = 1;
  }
}

void UpdateMultiDimReshape(Tensor& t, uint64_t* reshape, int dim_size) {
  for (size_t i = 0; i < t.shape().Dims(); ++i) {
    reshape[i] = t.shape().DimSize(i);
  }
  for (size_t i = t.shape().Dims(); i < dim_size; ++i) {
    reshape[i] = 1;
  }
}

/*
OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP(Sub, SubFunctor)
OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP(Mul, MulFunctor)
OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP(Div, DivFunctor)
OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP(Max, MaxFunctor)
OPENMI_REGISTER_BINARY_ELEMENT_WISE_OP(Min, MinFunctor)
*/
OPENMI_REGISTER_FILE_TAG(cwise_ops_binary);

}
