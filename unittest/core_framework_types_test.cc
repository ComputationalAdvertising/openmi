#include "openmi/pb/types.pb.h"
#include <string>
#include "base/logging.h"
#include <stdio.h>
#include <iostream>
#include "types.h"

using namespace openmi;
using namespace openmi::pb;
using namespace std;

template <typename T>
struct AccumulatorType {
  typedef T type;
};

std::string DataTypeString(DataType dtype) {
  switch (dtype) {
    case DT_INVALID:
      return "INVALID";
    case DT_FLOAT:
      return "float";
    case DT_DOUBLE:
      return "double";
    default:
      LOG(ERROR) << "Unrecognized DataType enum value: " << dtype;
      return std::string("unknown dtype enum(") + std::to_string(dtype) + ")";
  }
}

int main() {
  printf("DF_FLOAT: %s\n", DataTypeString(DT_FLOAT).c_str());
  printf("DF_DOUBLE: %s\n", DataTypeString(DT_DOUBLE).c_str());
  std::string r = DataTypeString(DT_INT8);
  printf("DT_INT8: %s\n", r.c_str());
  printf("size of DF_FLOAT: %d\n", SizeOfType(DT_FLOAT));
  printf("size of DF_DOUBLE: %d\n", SizeOfType(DT_DOUBLE));
  printf("size of DF_INT8_T: %d\n", SizeOfType(DT_INT8));

  printf("size of float: %d\n", sizeof(typename AccumulatorType<float>::type));
  printf("size of double: %d\n", sizeof(typename AccumulatorType<double>::type));

  printf("size of float: %d\n", sizeof(typename EnumToDataType<DT_FLOAT>::T));
  printf("size of double: %d\n", sizeof(EnumToDataType<DT_DOUBLE>::T));
  return 0;
}
