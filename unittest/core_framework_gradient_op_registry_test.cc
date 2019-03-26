#include "gradient_op_registry.h"
#include "base/logging.h"
#include "graph.h"
#include <memory>
#include <vector>
using namespace openmi;

int main(int argc, char** argv) {
  OpKernel* op;
  proto::NodeDef node_def;
  node_def.set_name("MatMul");
  node_def.set_op("MatMul");
  node_def.set_device("CPU");
  NodeInfo node_info(node_def, 1, NC_OP, NS_FORWARD);
  Node n;
  n.Initialize(node_info);
  LOG(INFO) << "before look up...";
  Status status = GradientOpRegistry::Instance().LookUp(n, &op);
  LOG(INFO) << "after look up...";
  if (op == nullptr) {
    LOG(ERROR) << "func is nullptr";
  }
  op->Compute(nullptr);
  return 0;
}
