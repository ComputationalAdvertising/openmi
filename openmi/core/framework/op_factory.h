#ifndef OPENMI_CORE_FRAMEWORK_OP_FACTORY_H_
#define OPENMI_CORE_FRAMEWORK_OP_FACTORY_H_ 

#include <functional>
#include "base/register.h"

using namespace openmi;

namespace openmi {

class Op;

class OpFactory: public openmi::FunctionRegisterBase<OpFactory, std::function<Op*()> > {
};

#define REGISTER_OP(ClassName) \
  OPENMI_REGISTER_OBJECT(::openmi::OpFactory, OpFactory, ClassName) \
  .SetFunction([]() { return new ClassName(); })

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_OP_FACTORY_H_
