#include "local_device.h"
#include "eigen_thread_pool.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "device_registry.h"

using namespace openmi;

namespace openmi {

// static defination
bool LocalDevice::use_global_threadpool_ = true;

struct LocalDevice::EigenThreadPoolInfo {
  explicit EigenThreadPoolInfo(int num_threads) {
    eigen_worker_threads_.num_threads = num_threads;
    eigen_worker_threads_.workers = new thread::EigenThreadPool(num_threads);
    eigen_device_.reset(
      new Eigen::ThreadPoolDevice(
        eigen_worker_threads_.workers->Get(), 
        eigen_worker_threads_.num_threads)
    );
  }

  ~EigenThreadPoolInfo() {
    eigen_device_.reset();
    delete eigen_worker_threads_.workers;
  }

  Device::CpuWorkerThreads eigen_worker_threads_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_device_;
}; // struct EigenThreadPoolInfo

LocalDevice::LocalDevice(int num_threads) 
  :  owned_tp_info_(nullptr) {
  LocalDevice::EigenThreadPoolInfo* tp_info;
  if (use_global_threadpool_) {
    // All ThreadPoolDevice in the process will use this single fixed size threadpool 
    // for numerial computations 
    static LocalDevice::EigenThreadPoolInfo* global_tp_info =
      new LocalDevice::EigenThreadPoolInfo(num_threads);
    tp_info = global_tp_info;
  } else {
    // Each LocalDevice owns a separte ThreadPoolDevice 
    owned_tp_info_.reset(new LocalDevice::EigenThreadPoolInfo(num_threads));
    tp_info = owned_tp_info_.get();
  }

  SetCpuWorkerThreads(& tp_info->eigen_worker_threads_);
  SetEigenCpuDevice(tp_info->eigen_device_.get());
}

LocalDevice::~LocalDevice() {
}

std::string LocalDevice::DeviceType() {
  return "CPU";
}

REGISTER_DEVICE(LocalDevice, CPU);

OPENMI_REGISTER_FILE_TAG(CPU);

} // namespace openmi
