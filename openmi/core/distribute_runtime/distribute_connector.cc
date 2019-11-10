#include "openmi/core/distribute_runtime/distribute_connector.h"

namespace openmi {

DistributeConnector::DistributeConnector(): thrift_client_(nullptr) {

}

DistributeConnector::~DistributeConnector() {

}

int DistributeConnector::Init(proto::comm::RpcComm& rpc_comm, int conn_timeout, int req_timeout) {
  auto rpc_type = rpc_comm.rpc_comm_type;
  switch (rpc_type) {
    case proto::comm::THRIFT: case proto::comm::SEASTAR: case proto::comm::GRPC:
    {
      std::vector<std::string> ips;
      std::vector<int> ports; 
      for (int i = 0; i < rpc_comm.ip_and_ports_size(); ++i) {
        auto ip_and_port = rpc_comm.ip_and_ports(i);
        ips.push_back(ip_and_port.ip());
        ports.push_back(ip_and_port.port());
      }
      thrift_client_ = std::make_shared<ThriftClientWrapper<PsClient>>(ips, ports, conn_timeout, req_timeout);
      if (thrift_client_ == nullptr) {
        LOG(ERROR) << "distribute connector init failed.";
        return -1;
      }
    }
    default: {
      LOG(ERROR) << "Unknown rpc comm type: " << proto::comm::RpcCommType_Name(rpc_type);
      return -1;
    }
  }

  return 0;
}

}