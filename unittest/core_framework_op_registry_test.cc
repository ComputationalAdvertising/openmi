#include "op_registry.h"
#include "base/logging.h"

int main(int argc, char** argv) {
  /*
  OpKernel* op_kernel;
  proto::NodeDef node_def;
  node_def.set_op("UnaryOp");
  node_def.set_device("CPU");

  OpRegistry::Instance().LookUp(node_def, &op_kernel);

  if (op_kernel == nullptr) {
    LOG(ERROR) << "op_kernel is nullptr";
    return 0;
  } 
  OpKernelContext::Params* params = new OpKernelContext::Params();
  OpKernelContext* context = new OpKernelContext(params);
  op_kernel->Compute(context);
  */
  return 0;
}
