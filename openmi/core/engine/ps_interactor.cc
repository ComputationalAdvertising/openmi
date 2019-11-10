#include "openmi/core/engine/ps_interactor.h"

namespace openmi {

int PsInteractor::Init(std::vector<std::string>& hosts, std::vector<int>& ports) {
  if (hosts.empty() || ports.empty()) {
    LOG(ERROR) << "list of host or port is empty. please check host|port config.";
    return -1;
  }
  if (hosts.size() != ports.size()) {
    LOG(ERROR) << "length between hosts and ports not match. "
               << hosts.size() << " vs " << ports.size();
    return -1;
  }

  ps_shard_num_ = (int) hosts.size();

  for (int i = 0; i < ps_shard_num_; ++i) {
    auto ps_sharding = std::make_shared<PsSharding>(name_, ps_slice_num_);
    if (ps_sharding == nullptr) {
      LOG(ERROR) << "ps sharding create failed.";
      return -1;
    }
    if (ps_sharding->Init(hosts, ports) != 0) {
      LOG(INFO) << "ps sharding init failed";
      return -1;
    }
    ps_shardings_.push_back(ps_sharding);
  }
  return 0;
}

std::string PsInteractor::Create(const std::string& gdef, bool is_binary) {
  std::vector<std::string> rsqs(ps_shard_num_);
  // TODO 多线程+同步
  for (size_t i = 0; i < ps_shard_num_; ++i) {
    ps_shardings_[i]->Create(rsqs[i], gdef, is_binary);
  }
  return rsqs[0];
}

// TODO 上游有多线程需加锁
void PsInteractor::AddPull(uint64_t fid, int field) {
  ps_shardings_[ShardId(fid)]->AddPull(fid, field);
}

// TODO 上游有多线程需加锁
void PsInteractor::AddPush(uint64_t fid, int field, std::vector<float>& values) {
  ps_shardings_[ShardId(fid)]->AddPush(fid, field, values);
}

void PsInteractor::Pull(std::string& value_type, const std::string& req_id) {
  // TODO 多线程+同步机制
  for (size_t i = 0; i < ps_shard_num_; ++i) {
    ps_shardings_[i]->Pull(value_type, req_id);
  }

  // TODO 多线程
  fid2paramdata_.clear();
  for (size_t i = 0; i < ps_shard_num_; ++i) {
    ps_shardings_[i]->PullRsqs(fid2paramdata_);
  }
}

void PsInteractor::Push(std::string& value_type, const std::string& req_id) {  
  // TODO 多线程+同步机制
  for (size_t i = 0; i < ps_shard_num_; ++i) {
    ps_shardings_[i]->Push(value_type, req_id);
  }
}

void PsInteractor::ClearPull() {
  for (size_t i = 0; i < ps_shardings_.size(); ++i) {
    ps_shardings_[i]->ClearPull();
  }
}

void PsInteractor::ClearPush() {
  for (size_t i = 0; i < ps_shardings_.size(); ++i) {
    ps_shardings_[i]->ClearPush();
  }
}

}