#ifndef OPENMI_CORE_COMMON_RUNTIME_LOCAL_DEVICE_H_
#define OPENMI_CORE_COMMON_RUNTIME_LOCAL_DEVICE_H_ 

#include <memory>
#include <string>
#include "device.h"

namespace openmi {

class LocalDevice : public Device {
public:
  LocalDevice(int num_threads = 2);

  virtual ~LocalDevice();

  std::string DeviceType() override;

  struct EigenThreadPoolInfo; 

  static void SetUseGlobalThreadPool(bool global_threadpool) {
    use_global_threadpool_ = global_threadpool;
  }

private:
  static bool use_global_threadpool_;
  std::unique_ptr<EigenThreadPoolInfo> owned_tp_info_;
  
  LocalDevice(const LocalDevice&);
  void operator=(const LocalDevice&);
};

} // namespace openmi
#endif // OPENMI_CORE_COMMON_RUNTIME_LOCAL_DEVICE_H_
