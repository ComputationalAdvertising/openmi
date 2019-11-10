#ifndef OPENMI_CORE_ENGINE_PS_SHARDING_H_
#define OPENMI_CORE_ENGINE_PS_SHARDING_H_

#include "openmi/core/distribute_runtime/thrift_client_wrapper.h"
#include "openmi/idl/proto/communication.pb.h"
using namespace openmi;
#include "openmi/gen-cpp/Ps.h"
using namespace openmi::thrift;

namespace openmi {

// multiply slice responds to a single ps
struct PsSlice {
  proto::comm::CommData pull_req_;
  proto::comm::CommData pull_rsq_;
  proto::comm::CommData push_req_;

  void ClearPull() {
    pull_req_.Clear();
    pull_rsq_.Clear();
  }

  void ClearPush() {
    push_req_.Clear();
  }
}; 

// a single ps
class PsSharding {
public: 
  PsSharding(std::string graph_name, int ps_slice_num = 8);

  ~PsSharding();

  int Init(std::vector<std::string>& hosts, std::vector<int>& ports);

  void Create(std::string& _return, const std::string& gdef, bool is_binary);

  void AddPull(uint64_t& fid, int field);

  void AddPush(uint64_t& fid, int field, std::vector<float>& values);

  void ClearPull();

  void ClearPush();

  void Pull(std::string& value_type, const std::string& req_id);

  void Push(std::string& value_type, const std::string& req_id);

  void PullRsqs(std::unordered_map<uint64_t, proto::comm::ValueList>& _return) {
    for (int i = 0; i < ps_slice_num_; ++i) {
      auto rsqs = ps_slices_[i].pull_rsq_;
      for (int k = 0; k < rsqs.keys_size(); ++k) {
        _return.insert({rsqs.keys(k), rsqs.vals(k)});
      }
    }
  }
  
private: 
  inline int SliceId(uint64_t fid) { 
    return (int) (fid % ps_slice_num_); 
  }

private:
  std::string graph_name_;
  std::shared_ptr<PsClient> thrift_client_;
  int ps_slice_num_;
  std::vector<PsSlice> ps_slices_;
  std::vector<std::string> reqs_;
  std::vector<std::string> rsqs_;
  // TODO 
  int conn_timeout = 500;
  int req_timeout = 500;
}; // class PsSharding

typedef std::shared_ptr<PsSharding> PsShardingPtr;

} // namespace openmi
#endif // OPENMI_CORE_ENGINE_PS_SHARDING_H_