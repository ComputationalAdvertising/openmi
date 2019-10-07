/*!
 *  Copyright (c) 2017 by Contributors
 *  \file thrift_server_wrapper.h
 *  \brief thrift server wrapper
 *  \author ZhouYong
 */
#ifndef OPENMI_BASE_THRIFT_SERVER_WRAPPER_H_
#define OPENMI_BASE_THRIFT_SERVER_WRAPPER_H_

#include <boost/shared_ptr.hpp>
//#include <glog/logging.h>
#include "logging.h"
#include <thrift/concurrency/ThreadManager.h>
#include <thrift/concurrency/PosixThreadFactory.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TNonblockingServer.h>
#include <string>

using namespace apache::thrift;
using namespace apache::thrift::concurrency;
using namespace apache::thrift::protocol;
using namespace apache::thrift::server;
using namespace apache::thrift::transport;

namespace openmi {

class ThriftServerWrapper {
public:
  ThriftServerWrapper(boost::shared_ptr<TProcessor> processor, 
                      const int port, 
                      const int num_thread, 
                      const int num_io_thread) : 
  processor_(processor), port_(port), num_thread_(num_thread), num_io_thread_(num_io_thread) {
    InitTNonblockingServer();
    LOG(INFO) << "InitTNonblockingServer port: " << port_ 
              << ", num_thread: " << num_thread << ", num_io_thread: " << num_io_thread_;
  }

  inline std::shared_ptr<TNonblockingServer> GetThriftServer() { return server_; }
  
  inline boost::shared_ptr<ThreadManager> GetThreadManager() { return thread_mgr_; }

private:
  void InitTNonblockingServer() {
    boost::shared_ptr<TProtocolFactory> protocol_factory(new TBinaryProtocolFactory());
    boost::shared_ptr<PosixThreadFactory> thread_factory(new PosixThreadFactory());
    thread_mgr_ = boost::shared_ptr<ThreadManager>(ThreadManager::newSimpleThreadManager(num_thread_));
    thread_mgr_->threadFactory(thread_factory);
    thread_mgr_->start(); 
    server_ = std::make_shared<TNonblockingServer>(processor_, protocol_factory, port_, thread_mgr_);
    server_->setNumIOThreads(num_io_thread_);
  }

private:
  boost::shared_ptr<ThreadManager> thread_mgr_;
  boost::shared_ptr<TProcessor> processor_;
  std::shared_ptr<TNonblockingServer> server_;
  int port_;
  int num_thread_;
  int num_io_thread_;
};

} // namespace
#endif // OPENMI_BASE_THRIFT_SERVER_WRAPPER_H_
