#ifndef OPENMI_CORE_ENGINE_PS_INTERACTOR_H_
#define OPENMI_CORE_ENGINE_PS_INTERACTOR_H_

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "base/logging.h"
#include "openmi/idl/proto/communication.pb.h"
#include "openmi/core/engine/ps_sharding.h"
using namespace openmi;

namespace openmi {

class PsInteractor {
public: 
  PsInteractor(const std::string& name): name_(name) {}

  ~PsInteractor() {}

  int Init(std::vector<std::string>& hosts, std::vector<int>& ports);

  std::string Create(const std::string& gdef, bool is_binary);

  void AddPull(uint64_t fid, int field);

  void AddPush(uint64_t fid, int field, std::vector<float>& values);

  void ClearPull();

  void ClearPush();

  void Pull(std::string& value_type, const std::string& req_id);

  void Push(std::string& value_type, const std::string& req_id);

  std::unordered_map<uint64_t, proto::comm::ValueList>& GetFid2ParamData() {
    return fid2paramdata_;
  }

private: 
  inline int ShardId(uint64_t fid) { 
    return (int) (fid % ps_shard_num_); 
  }

private:
  std::string name_;
  int ps_shard_num_;
  std::vector<PsShardingPtr> ps_shardings_;
  std::unordered_map<uint64_t, proto::comm::ValueList> fid2paramdata_;
  int ps_slice_num_ = 8;
}; // class PsAccessor
typedef std::shared_ptr<PsInteractor> PsInteractorPtr;

} // namespace openmi

#endif // OPENMI_CORE_ENGINE_PS_INTERACTOR_H_