#ifndef OPENMI_CORE_FRAMEWORK_DEVICE_H_
#define OPENMI_CORE_FRAMEWORK_DEVICE_H_ 

#include <unsupported/Eigen/CXX11/Tensor>
#include "base/allocator.h"

namespace openmi {

namespace thread {
class EigenThreadPool;
}

class Device {
public:
  Device() {}

  virtual ~Device() {}

  void set_allocator(Allocator* allocator) {
    allocator_.reset(allocator);
  }

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

  const Eigen::ThreadPoolDevice& eigen_cpu_device() const {
    CHECK(eigen_cpu_device_ != nullptr) 
      << "eigen_cpu_device_ == nullptr";
    return *eigen_cpu_device_;
  }

private:
  std::unique_ptr<Allocator> allocator_;
  CpuWorkerThreads* cpu_worker_threads_ = nullptr;
  Eigen::ThreadPoolDevice* eigen_cpu_device_ = nullptr;
}; // class Device

typedef Eigen::ThreadPoolDevice CpuDevice;
typedef Eigen::GpuDevice GpuDevice;

} // namespace openmi
#endif // OPENMI_CORE_FRAMEWORK_DEVICE_H_
