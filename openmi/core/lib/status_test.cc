#include "core/lib/status.h"

using namespace openmi;

void status_test() {
  if (Status::OK().code() != OK) {
    printf("OK().code() != OK\n");
  } else {
    printf("OK().code() == OK\n");
  }
}

int main(int argc, char** argv) {
  status_test();
  return 0;
}
