#ifndef OPENMI_CORE_KERNELS_MATMUL_OP_H_
#define OPENMI_CORE_KERNELS_MATMUL_OP_H_ 

#include "op_kernel.h"

namespace openmi {

template <typename Device, typename TType, typename DimPair>
void MatMulImpl(const Device& d, TType out, TType in0, TType in1, const DimPair& dim_pair) {
  out.device(d) = in0.contract(in1, dim_pair);
};

} // namespace openmi
#endif // OPENMI_CORE_KERNELS_MATMUL_OP_H_
