#include "openmi/idl/proto/types.pb.h"
#include <string>
#include "base/logging.h"
#include <stdio.h>
#include <iostream>
#include "types.h"

using namespace openmi;
using namespace openmi::proto;
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
  printf("DF_FLOAT: %s\n", DataType_Name(DT_FLOAT).c_str());
  printf("DF_DOUBLE: %s\n", DataTypeString(DT_DOUBLE).c_str());
  std::string r = DataTypeString(DT_INT8);
  printf("DT_INT8: %s\n", r.c_str());
  printf("size of DF_FLOAT: %zu\n", SizeOfType(DT_FLOAT));
  printf("size of DF_DOUBLE: %zu\n", SizeOfType(DT_DOUBLE));
  printf("size of DF_INT8_T: %zu\n", SizeOfType(DT_INT8));

  printf("size of float: %lu\n", sizeof(typename AccumulatorType<float>::type));
  printf("size of double: %lu\n", sizeof(typename AccumulatorType<double>::type));

  printf("size of float: %lu\n", sizeof(typename EnumToDataType<DT_FLOAT>::T));
  printf("size of double: %lu\n", sizeof(EnumToDataType<DT_DOUBLE>::T));
  return 0;
}
