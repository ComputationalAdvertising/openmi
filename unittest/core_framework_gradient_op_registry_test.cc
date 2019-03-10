#include "gradient_op_registry.h"
#include "base/logging.h"
using namespace openmi;

Status UnaryOpGradient() {
  LOG(INFO) << "UnaryOpGradient ...";
  return Status::OK();
}

OPENMI_REGISTER_GRADIENT_OP(UnaryOp, UnaryOpGradient);

int main(int argc, char** argv) {
  GradFunc* func = nullptr;
  std::string name("UnaryOp");
  Status status = GradientOpRegistry::Instance().LookUp(name, &func);
  if (func == nullptr) {
    LOG(ERROR) << "func is nullptr";
  }
  status = (*func)();
  return 0;
}
