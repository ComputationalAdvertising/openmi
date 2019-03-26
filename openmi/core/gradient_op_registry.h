#ifndef OPENMI_CORE_GRADIENT_OP_REGISTRY_H_
#define OPENMI_CORE_GRADIENT_OP_REGISTRY_H_ 

#include <unordered_map>
#include "base/singleton.h"
#include "status.h"
#include "op_registry.h"
#include "op_kernel.h"

namespace openmi {

class GradientOpRegistry : public Singleton<GradientOpRegistry> {
public:
  void RegisterOp(std::string name, OpRegistrationEntry* entry);

  Status LookUp(Node& node, OpKernel** grad_op);

private:
  std::unordered_map<std::string, OpRegistrationData> gradient_op_mapper_;
}; // class GradientOpRegistry 

class GradientOpRegistryReceiver {
public:
  GradientOpRegistryReceiver(const OpRegistryHelper& helper) {
    GradientOpRegistry::Instance().RegisterOp(helper.Name(), helper.Entry());
  }
}; // class GradientOpRegistryReceiver 

#define OPENMI_REGISTER_GRADIENT_KERNEL_UNIQ_HELPER(name, ctr) \
  static ::openmi::GradientOpRegistryReceiver register_gradient_op_## ctr ## _ ##name \
  __attribute__((unused)) = ::openmi::OpRegistryHelper(#name)

#define OPENMI_REGISTER_GRADIENT_KERNEL_UNIQ(name, ctr) \
  OPENMI_REGISTER_GRADIENT_KERNEL_UNIQ_HELPER(name, ctr)

#define OPENMI_REGISTER_GRADIENT_OP_KERNEL(name, ...) \
  OPENMI_REGISTER_GRADIENT_KERNEL_UNIQ(name, __COUNTER__).op_kernel<__VA_ARGS__>()

} // namespace openmi 
#endif // OPENMI_CORE_GRADIENT_OP_REGISTRY_H_
