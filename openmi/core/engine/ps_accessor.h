#ifndef OPENMI_CORE_ENGINE_PS_ACCESSOR_H_
#define OPENMI_CORE_ENGINE_PS_ACCESSOR_H_

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "base/logging.h"
#include "openmi/idl/proto/engine.pb.h"
using namespace openmi;

namespace openmi {

class PsAccessor {
public: 
  PsAccessor() {}
  ~PsAccessor() {}

  void ClearPull() {
    pull_req_.Clear();
    pull_rsq_.Clear();
    fid2paramdata_.clear();
  }

  void ClearPush() {
    push_req_.Clear();
  }

  void AddPullFid(uint64_t fid, int column_id = -1) {
    pull_req_.add_keys(fid);
    pull_req_.add_columns(column_id);
  }

  void AddGradient(uint64_t fid, std::vector<float>& grads) {
    push_req_.add_keys(fid);
    auto* val_list = push_req_.add_vals();
    for (int i = 0; i < grads.size(); ++i) {
      val_list->add_val(grads.at(i));
    }
  }

  int PreparePull() {
    // TODO ps shard
    LOG(INFO) << __FUNCTION__ << " pulled fids:\n" << pull_req_.DebugString();
    return 0;
  }

  int Pull() {
    return 0;
  }

  int PreparePush() {
    // TODO ps shard
    LOG(INFO) << __FUNCTION__ << " pushed gradients:\n" << push_req_.DebugString();
    return 0;
  }

  int Push() {
    return 0;
  }

  std::unordered_map<uint64_t, proto::internal::ValList>& GetFid2ParamData() {
    return fid2paramdata_;
  }

private:
  std::unordered_map<uint64_t, proto::internal::ValList> fid2paramdata_;
  proto::internal::ModelParamData pull_req_;
  proto::internal::ModelParamData pull_rsq_;
  proto::internal::ModelParamData push_req_;
}; // class PsAccessor

} // namespace openmi

#endif // OPENMI_CORE_ENGINE_PS_ACCESSOR_H_