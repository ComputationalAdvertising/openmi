#ifndef OPENMI_CORE_FRAMEWORK_DEVICE_FACTORY_H_
#define OPENMI_CORE_FRAMEWORK_DEVICE_FACTORY_H_ 

#include "device.h"
#include <functional>
#include "base/register.h"

namespace openmi {

class DeviceFactory : public openmi::FunctionRegisterBase<DeviceFactory, std::function<Device*()> > {
};

#define REGISTER_DEVICE(Device, name) \
  OPENMI_REGISTER_OBJECT_HELPER(::openmi::DeviceFactory, DeviceFactory, Device, name) \
  .SetFunction([]() { return new Device(); })

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_DEVICE_FACTORY_H_ 
