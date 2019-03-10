#ifndef OPENMI_CORE_GRADIENT_OP_REGISTRY_H_
#define OPENMI_CORE_GRADIENT_OP_REGISTRY_H_ 

#include <unordered_map>
#include "base/singleton.h"
#include "status.h"

namespace openmi {

typedef Status (*GradFunc)();

class GradientOpRegistry : public Singleton<GradientOpRegistry> {
public:
  bool Register(std::string name, GradFunc grad_func);

  Status LookUp(std::string name, GradFunc** grad_op);

private:
  std::unordered_map<std::string, GradFunc> gradient_op_mapper_;
}; // class GradientOpRegistry 

#define OPENMI_REGISTER_GRADIENT_OP(name, func) \
  OPENMI_REGISTER_GRADIENT_OP_UNIQ_HELPER(__COUNTER__, name, func)

#define OPENMI_REGISTER_NO_GRADIENT_OP(name) \
  OPENMI_REGISTER_GRADIENT_OP_UNIQ_HELPER(__COUNTER__, name, nullptr)
  
#define OPENMI_REGISTER_GRADIENT_OP_UNIQ_HELPER(ctr, name, func)  \
  static bool register_grad_op_##name = \
    ::openmi::GradientOpRegistry::Instance().Register(#name, func)

} // namespace openmi 
#endif // OPENMI_CORE_GRADIENT_OP_REGISTRY_H_
