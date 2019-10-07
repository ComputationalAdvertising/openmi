/*!
 *  Copyright (c) 2017 by Contributors
 *  \file thrift_client_wrapper.h
 *  \brief thrift client wrapper
 */
#ifndef OPENMI_BASE_THRIFT_CLIENT_WRAPPER_H_
#define OPENMI_BASE_THRIFT_CLIENT_WRAPPER_H_

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include "logging.h"
#include <thrift/transport/TSocketPool.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/protocol/TBinaryProtocol.h>

using namespace apache::thrift;
using namespace apache::thrift::transport;
using namespace apache::thrift::protocol;

namespace openmi {

template<class Client>
class ThriftClientWrapper {
public:
  ThriftClientWrapper(const std::vector<std::string>& hosts, 
                      const std::vector<int>& ports, 
                      const int conn_timeout = 0, 
                      int timeout = 0): conn_timeout_(conn_timeout), timeout_(timeout) {
    InitThriftClient(hosts, ports);
  }

  ThriftClientWrapper(const std::string& host, 
                      const int port, 
                      const int conn_timeout = 0, 
                      int timeout = 0) : conn_timeout_(conn_timeout), timeout_(timeout) {
    InitThriftClient(host, port);
  }

  ~ThriftClientWrapper() {}

  inline std::shared_ptr<Client> GetThriftClient() { return client_; }

public:
  void InitThriftClient(const std::vector<std::string>& hosts, const std::vector<int>& ports) {
    boost::shared_ptr<TSocket> socket;
    if (hosts.empty() || ports.empty()) {
      throw std::runtime_error("hosts or port invalid. please check it.");
    }
    if (hosts.size() > 1 && ports.size() > 1) {
      socket.reset(new TSocketPool(hosts, ports));
    } else if (hosts.size() > 1 && ports.size() == 1) {
      std::vector<int> ports_(ports[0], hosts.size());
      socket.reset(new TSocketPool(hosts, ports_));
    } else if (hosts.size() == 1 && ports.size() == 1) {
      socket.reset(new TSocket(hosts[0], ports[0]));
    } 

    if (conn_timeout_) {
      socket->setConnTimeout(conn_timeout_);
    } else if (timeout_) {
      socket->setConnTimeout(timeout_);
    }

    if (timeout_) {
      socket->setSendTimeout(timeout_);
      socket->setRecvTimeout(timeout_);
    }
    boost::shared_ptr<TTransport> transport(new TFramedTransport(socket));
    boost::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));

    client_.reset(new Client(protocol));
  }

  void InitThriftClient(const std::string& host, const int port) {
    boost::shared_ptr<TSocket> socket(new TSocket(host, port));

    if (conn_timeout_) {
      socket->setConnTimeout(conn_timeout_);
    } else if (timeout_) {
      socket->setConnTimeout(timeout_);
    }

    if (timeout_) {
      socket->setSendTimeout(timeout_);
      socket->setRecvTimeout(timeout_);
    }

    boost::shared_ptr<TTransport> transport(new TFramedTransport(socket));
    boost::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));

    client_.reset(new Client(protocol));
  }

private:
  std::shared_ptr<Client> client_;
  int conn_timeout_;
  int timeout_;
};

} // namespace
#endif // OPENMI_BASE_THRIFT_CLIENT_WRAPPER_H_
