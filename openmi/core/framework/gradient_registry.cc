#include "gradient_registry.h"

namespace openmi {

bool GradientRegistry::Register(const std::string& op, GradConstructor creator) {
  LOG(DEBUG) << "GradientRegistry op:" << op;
  auto it = grad_mapper_.find(op);
  CHECK(it == grad_mapper_.end()) 
    << op << " has already registered.";
  grad_mapper_.insert({op, creator});
  return true;
}

void GradientRegistry::Lookup(const std::string& op, GradConstructor* creator) {
  auto it = grad_mapper_.find(op);
  CHECK(it != grad_mapper_.end()) 
    << "op '" << op << "' not in gradient registry.";
  *creator = it->second;
}

OPENMI_REGISTER_LINK_TAG(math_grad_constructor);

}
