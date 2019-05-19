#ifndef OPENMI_CORE_COMMON_RUNTIME_EIGEN_THREAD_POOL_H_
#define OPENMI_CORE_COMMON_RUNTIME_EIGEN_THREAD_POOL_H_ 

#include <assert.h>
#include <functional>
#include <unsupported/Eigen/CXX11/ThreadPool>

namespace Eigen {
template <typename Env> using ThreadPoolTempl = NonBlockingThreadPoolTempl<Env>;
typedef NonBlockingThreadPool ThreadPool;
} // endn namespace Eigen

namespace openmi {
namespace thread {

// eigen thread pool
class EigenThreadPool {
public:
  explicit EigenThreadPool(int num_threads = 2)
    : pool_(num_threads), num_threads_(num_threads) {}
  
  void Schedule(std::function<void()> func) {
    pool_.Schedule(std::move(func));
  }

  Eigen::ThreadPool* Get() {
    return &pool_;
  }

  int NumThreads() const {
    return num_threads_;
  }

  int CurrentThreadId() const {
    return pool_.CurrentThreadId();
  }
private:
  Eigen::ThreadPool pool_;
  int num_threads_;
}; // class EigenThreadPoolWrapper

} // namespace thread
} // namespace openmi
#endif // OPENMI_CORE_COMMON_RUNTIME_EIGEN_THREAD_POOL_H_
