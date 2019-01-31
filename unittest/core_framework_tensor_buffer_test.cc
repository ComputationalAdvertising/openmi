#include "core/framework/tensor_buffer.h"
#include <vector>
#include "base/logging.h"

using namespace openmi;

int main(int argc, char** argv) {
  std::vector<float> v(10);
  for (auto i = 0; i < v.size(); ++i) {
    v[i] = (i+1) * 0.1f;
  }

  TensorBuffer<float>* tb = new TensorBuffer<float>(v.data(), v.size());
  LOG(INFO) << "size: " << tb->Size();
  CHECK_EQ(v.size(), tb->Size());

  TensorBuffer<double>* tbd = new TensorBuffer<double>(v.size());
  LOG(INFO) << "memory_bytes_size: " << tbd->MemoryBytes();

  delete tb;
  delete tbd;
  return 0;
}
