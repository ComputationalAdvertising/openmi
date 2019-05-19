#ifndef OPENMI_CORE_FRAMEWORK_GRADIENT_REGISTRY_H_
#define OPENMI_CORE_FRAMEWORK_GRADIENT_REGISTRY_H_ 

#include "base/register.h"
#include "base/singleton.h"
#include "graph.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace openmi {

typedef void (*GradConstructor)(Node& node, 
                                std::vector<Node*>& dy_list, 
                                std::vector<Node*>& dx_list, 
                                Graph& g);

/*!
 * \brief Register gradient constructor. not gradient op
 */
class GradientRegistry : public openmi::Singleton<GradientRegistry> {
public:
  bool Register(const std::string& op, GradConstructor creator);

  void Lookup(const std::string& op, GradConstructor* creator);

private:
  std::unordered_map<std::string, GradConstructor> grad_mapper_;
}; // class GradientRegistry
} // namespace openmi

// Macros used to define gradient constructor functions for ops.
#define OPENMI_REGISTER_GRADIENT(name, fn) \
  OPENMI_REGISTER_GRADIENT_UNIQ_HELPER(__COUNTER__, name, fn)

#define OPENMI_REGISTER_GRADIENT_UNIQ_HELPER(ctr, name, fn) \
  OPENMI_REGISTER_GRADIENT_UNIQ(ctr, name, fn)

#define OPENMI_REGISTER_GRADIENT_UNIQ(ctr, name, fn) \
  static bool gradient_creator_##ctr##_name = \
    ::openmi::GradientRegistry::Instance().Register(#name, fn)

#endif // OPENMI_CORE_FRAMEWORK_GRADIENT_REGISTRY_H_
