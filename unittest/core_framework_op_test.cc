#include "core/framework/op.h"
#include <string>

int main() {
  Op* add_op = openmi::Register<OpFactory>::Find("AddOp")->func();
  LOG(INFO) << "add_op, name: " << add_op->Name();
  
  Op* multiply_op = openmi::Register<OpFactory>::Find("MultiplyOp")->func();
  LOG(INFO) << "multiply_op, name: " << multiply_op->Name();

  delete add_op; delete multiply_op;

  return 0;
}
