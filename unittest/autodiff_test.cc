#include "variable.h"
#include "base/logging.h"

using namespace openmi;

int main(int argc, char** argv) {
  //Node x2 = Variable::Get("x2");
  //Node x3 = Variable::Get("x3");

  Node x2("x2", 0);
  Node x3("x3", 0);

  Node y = x2 * x3 + x2 + x3;
  
  LOG(INFO) << "y: " << y.DebugString();

  return 0;
}
