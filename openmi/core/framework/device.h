#ifndef OPENMI_BASE_DEVICE_H_
#define OPENMI_BASE_DEVICE_H_ 

#include <unsupported/Eigen/CXX11/Tensor>
#include "base/allocator.h"

namespace openmi {

namespace thread {
class EigenThreadPool;
}

class Device {
public:
  Device(Allocator* allocator) {
    allocator_.reset(allocator);
  }

  virtual ~Device() {}

  Allocator* GetAllocator() {
    return allocator_.get();
  }

  virtual std::string DeviceType() = 0;

  struct CpuWorkerThreads {
    int num_threads = 0;
    thread::EigenThreadPool* workers = nullptr;
  };

  void SetCpuWorkerThreads(CpuWorkerThreads* cwt) {
    cpu_worker_threads_ = cwt;
  }

  const CpuWorkerThreads* GetCpuWorkerThreads() const {
    CHECK(cpu_worker_threads_ != nullptr);
    return cpu_worker_threads_;
  }

  void SetEigenCpuDevice(Eigen::ThreadPoolDevice* d) {
    eigen_cpu_device_ = d;
  }

private:
  std::unique_ptr<Allocator> allocator_;
  CpuWorkerThreads* cpu_worker_threads_ = nullptr;
  Eigen::ThreadPoolDevice* eigen_cpu_device_ = nullptr;
}; // class Device

} // namespace openmi
#endif // OPENMI_BASE_DEVICE_H_
