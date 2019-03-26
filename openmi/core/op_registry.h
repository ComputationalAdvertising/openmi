#ifndef OPENMI_CORE_FRAMEWORK_OP_REGESTRY_H_
#define OPENMI_CORE_FRAMEWORK_OP_REGESTRY_H_ 

#include "op_kernel.h"

#include <functional>
#include <set>
#include <string>
#include <unordered_map>

#include "base/register.h"
#include "base/singleton.h"
#include "graph.h"
#include "types.h"

using namespace openmi;

namespace openmi {

class OpKernel;

struct OpRegistrationEntry {
  OpRegistrationEntry() 
    : device("CPU"), allow_type(DT_FLOAT) {}
  std::string device;
  DataType allow_type;
  std::function<OpKernel*()> body;
};

// multi-device op kernel
class OpRegistrationData {
public:
  OpRegistrationEntry* FindEntry(
    const std::string device, DataType type);

  std::set<OpRegistrationEntry*> entrys;
  // TODO extra info
};

class OpRegistry : public Singleton<OpRegistry> {
public:
  void RegisterOp(std::string name, OpRegistrationEntry* entry);
  
  // TODO void --> status
  Status LookUp(Node& node, OpKernel** op_kernel);

private:
  std::unordered_map<std::string, OpRegistrationData> op_kernel_mapper_; 
}; 

class OpRegistryHelper {
public:
  explicit OpRegistryHelper(std::string op_name);

  template <typename T>
  OpRegistryHelper& op_kernel();
  
  OpRegistryHelper& SetBody(const std::function<OpKernel*()>& body);
  OpRegistryHelper& Device(std::string device);
  OpRegistryHelper& TypeConstraint(DataType type);

  template <typename T>
  OpRegistryHelper& TypeConstraint();
  
  std::string Name() const { return name_; };
  OpRegistrationEntry* Entry() const { return entry_; }

private:
  std::string name_;
  OpRegistrationEntry* entry_;
};

template <typename T>
OpRegistryHelper& OpRegistryHelper::op_kernel() {
  return SetBody([]() -> OpKernel* { return new T; });
}

template <typename T>
OpRegistryHelper& OpRegistryHelper::TypeConstraint() {
  return TypeConstraint(DataTypeToEnum<T>::v());
}

class OpRegistryReceiver {
public:
  OpRegistryReceiver(const OpRegistryHelper& helper) {
    OpRegistry::Instance().RegisterOp(helper.Name(), helper.Entry());
  }
}; 
  
} // namespace openmi 

#define OPENMI_REGISTER_SINGLE_ARGS(...) __VA_ARGS__

#define OPENMI_REGISTER_KERNEL_UNIQ_HELPER(name, ctr) \
  static ::openmi::OpRegistryReceiver register_op_## ctr ## _ ##name \
  __attribute__((unused)) = ::openmi::OpRegistryHelper(#name)

#define OPENMI_REGISTER_KERNEL_UNIQ(name, ctr) \
  OPENMI_REGISTER_KERNEL_UNIQ_HELPER(name, ctr)

#define OPENMI_REGISTER_OP_KERNEL(name, ...) \
  OPENMI_REGISTER_KERNEL_UNIQ(name, __COUNTER__).op_kernel<__VA_ARGS__>() 

#endif // OPENMI_CORE_FRAMEWORK_OP_REGESTRY_H_
