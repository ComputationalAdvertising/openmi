#ifndef OPENMI_CORE_FRAMEWORK_TYPES_H_
#define OPENMI_CORE_FRAMEWORK_TYPES_H_ 

#include "openmi/pb/types.pb.h"
using namespace openmi::pb;

namespace openmi {

template <class T>
struct DataTypeToEnum {};

template <DataType VALUE> 
struct EnumToDataType {};

#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)       \
  template <>                                 \
  struct DataTypeToEnum<TYPE> {               \
    static DataType v() { return DataType::ENUM; }      \
    static constexpr DataType value = DataType::ENUM;   \
  };                                          \
  template <>                                 \
  struct EnumToDataType<ENUM> {               \
    typedef TYPE T;                        \
  };

MATCH_TYPE_AND_ENUM(float, DT_FLOAT);
MATCH_TYPE_AND_ENUM(double, DT_DOUBLE);
MATCH_TYPE_AND_ENUM(int8_t, DT_INT8);
MATCH_TYPE_AND_ENUM(int16_t, DT_INT16);
MATCH_TYPE_AND_ENUM(int32_t, DT_INT32);
MATCH_TYPE_AND_ENUM(int64_t, DT_INT64);
MATCH_TYPE_AND_ENUM(bool, DT_BOOL);

#undef MATCH_TYPE_AND_ENUM 

#define PRINT(X) #X

#define DIRECT_ARG(...) __VA_ARGS__

#define OPENMI_TYPE_CASE(TYPE, STMTS)                                 \
    case ::openmi::DataTypeToEnum<TYPE>::value: {                     \
        printf("type: %s, stmts: %s\n", PRINT(TYPE), PRINT(STMTS));   \
        typedef TYPE T;                                               \
        STMTS;                                                        \
        break;                                                        \
    }

#define OPENMI_TYPE_CASES(TYPE_ENUM, STMTS)                 \
    switch (TYPE_ENUM) {                                    \
        OPENMI_TYPE_CASE(int8_t, DIRECT_ARG(STMTS))         \
        OPENMI_TYPE_CASE(int16_t, DIRECT_ARG(STMTS))        \
        OPENMI_TYPE_CASE(int32_t, DIRECT_ARG(STMTS))        \
        OPENMI_TYPE_CASE(int64_t, DIRECT_ARG(STMTS))        \
        OPENMI_TYPE_CASE(float, DIRECT_ARG(STMTS))          \
        OPENMI_TYPE_CASE(double, DIRECT_ARG(STMTS))         \
        OPENMI_TYPE_CASE(bool, DIRECT_ARG(STMTS))           \
      default: printf("xxx"); \
    }

    //default: OPENMI_CHECK(false) << "type error"; 

inline size_t SizeOfType(DataType type) {
  size_t ret = 0;
  OPENMI_TYPE_CASES(type, ret = sizeof(T));
  return ret;
}

} // namespace openmi 
#endif // OPENMI_CORE_FRAMEWORK_TYPES_H_
