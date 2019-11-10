#include "openmi/core/engine/ps_sharding.h"
#include "base/timer.h"
using namespace openmi;

namespace openmi {

PsSharding::PsSharding(std::string graph_name, int ps_slice_num)
  : graph_name_(graph_name), ps_slice_num_(ps_slice_num) {
    ps_slices_.resize(ps_slice_num_);
}

PsSharding::~PsSharding() {}

int PsSharding::Init(std::vector<std::string>& hosts, std::vector<int>& ports) {
  CHECK(hosts.size() == ports.size()) << "size beween hosts and ports not match. "
    << hosts.size() << " vs " << ports.size();
  
  std::shared_ptr<ThriftClientWrapper<PsClient> > wrapper = 
    std::make_shared<ThriftClientWrapper<PsClient> >(hosts, ports, conn_timeout, req_timeout);
  if (wrapper == nullptr) {
    LOG(ERROR) << "thrift wrapper create failed.";
    return -1;
  }
  thrift_client_ = wrapper->GetThriftClient();
  thrift_client_->getInputProtocol()->getTransport()->open();

  return 0;
}

void PsSharding::Create(std::string& _return, const std::string& gdef, bool is_binary) {
  Timer time;
  thrift_client_->Create(_return, gdef, is_binary);
  LOG(INFO) << __FUNCTION__ << ". time: " << time.Elapsed();
}

void PsSharding::AddPull(uint64_t& fid, int field) {
  auto& req = ps_slices_[SliceId(fid)].pull_req_;
  req.add_keys(fid);
  req.add_fields(field);
}

void PsSharding::AddPush(uint64_t& fid, int field, std::vector<float>& values) {
  auto& req = ps_slices_[SliceId(fid)].push_req_;
  req.add_keys(fid);
  req.add_fields(field);
  auto vals = req.add_vals();
  // TODO 优化性能 vector -> proto
  for (size_t i = 0; i < values.size(); ++i) {
    vals->add_val(values[i]);
  }
}

void PsSharding::ClearPull() {
  for (size_t i = 0; i < ps_slice_num_; ++i) {
    ps_slices_[i].ClearPull();
  }
}

void PsSharding::ClearPush() {
  for (size_t i = 0; i < ps_slice_num_; ++i) {
    ps_slices_[i].ClearPush();
  }
}

void PsSharding::Pull(std::string& value_type, const std::string& req_id) {
  reqs_.clear();
  reqs_.resize(ps_slice_num_);
  rsqs_.clear();
  rsqs_.resize(ps_slice_num_);
  
  // TODO parallel
  for (size_t i = 0; i < ps_slice_num_; ++i) {
    ps_slices_[i].pull_req_.SerializeToString(&reqs_[i]);
  }
  
  // TODO retry rpc && ack
  thrift_client_->Pull(rsqs_, graph_name_, reqs_, value_type, req_id);

  // TODO parallel
  for (size_t i = 0; i < ps_slice_num_; ++i) {
    ps_slices_[i].pull_rsq_.ParseFromString(rsqs_[i]);
  }
}

void PsSharding::Push(std::string& value_type, const std::string& req_id) {
  reqs_.clear();
  reqs_.resize(ps_slice_num_);

  // TODO parallel 
  for (size_t i = 0; i < ps_slice_num_; ++i) {
    ps_slices_[i].push_req_.SerializeToString(&reqs_[i]);
    LOG(INFO) << "i:" << i << ", push req:\n" << ps_slices_[i].push_req_.DebugString();
  }

  // TODO retry rpc && ack
  thrift_client_->Push(graph_name_, reqs_, value_type, 0, req_id);
}

} // namespace openmi