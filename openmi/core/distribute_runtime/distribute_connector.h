#ifndef OPENMI_CORE_DISTRIBUTE_RUNTIME_DISTRIBUTE_CONNECTOR_H_
#define OPENMI_CORE_DISTRIBUTE_RUNTIME_DISTRIBUTE_CONNECTOR_H_

#include <string>
#include <vector>
#include "openmi/idl/proto/communication.pb.h"
#include "openmi/core/distribute_runtime/thrift_client_wrapper.h"
using namespace openmi;
#include "openmi/gen-cpp/Ps.h"
using namespace openmi::thrift;

namespace openmi {

class DistributeConnector {
public: 
  DistributeConnector();

  ~DistributeConnector();

  int Init(proto::comm::RpcComm& rpc_comm, int conn_timeout, int req_timeout);

private:
  std::shared_ptr<ThriftClientWrapper<PsClient> > thrift_client_;
}; // class DistributeConnector

} // namespace openmi
#endif // OPENMI_CORE_DISTRIBUTE_RUNTIME_DISTRIBUTE_CONNECTOR_H_