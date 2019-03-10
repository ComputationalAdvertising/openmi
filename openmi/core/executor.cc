#include "executor.h"
#include "algorithm.h"
#include "device.h"
#include "device_registry.h"
#include "base/register.h"

using namespace openmi;

namespace openmi {

Executor::Executor(proto::GraphDef& gdef) {
  LOG(INFO) << "Executor construct ...";
  Status status = ConvertGraphDefToGraph(&gdef, &g_);
  LOG(INFO) << "Executor construct done ...";
  for (auto& kv: g_.node_mapper()) {
    DataType type = DT_FLOAT;
    auto it = kv.second->attrs().find("type");
    if (it != kv.second->attrs().end()) {
      CHECK(it->second.attr_type == ::openmi::AttrValue::kType);
      type = it->second.type;
    }

    Tensor* tensor;
    it = kv.second->attrs().find("shape");
    if (it != kv.second->attrs().end() && 
        it->second.attr_type == ::openmi::AttrValue::kShape) {
      tensor = new Tensor(type, it->second.shape);
    } else {
      tensor = new Tensor(type);
    }
    LOG(INFO) << "name: " << kv.second->def().name() 
      << ", is init: " << tensor->IsInitialized();
    if (tensor->IsInitialized()) {
      LOG(INFO) << "tensor shape: " << tensor->shape().DebugString();
    }
    session_state_.AddTensor(kv.second->def().name(), tensor);
  }

  TopoOrderList(g_.nodes(), g_.forward_topo_nodes(), &g_);

  // TODO CHECK DAG

  LOG(INFO) << "Executor init successfully.";
}

Executor::~Executor() {
}

Status Executor::Run() {
  for (auto& node: g_.forward_topo_nodes()) {
    LOG(INFO) << "node.name: " << node->def().name();
    OpKernelConstruction okc(node->def().name(), node->attrs());
    node->op()->Initialize(&okc);

    OpKernelContext::Params params;
    
    auto device_name = node->def().device();
    Device* device = openmi::Register<DeviceFactory>::Find(device_name)->func();
    CHECK(device != nullptr) << "deivce not exist. device:" << device_name;
    params.device = device;

    params.session_state = &session_state_;
    params.op_kernel = node->op();
    params.node_def = &node->def();
    params.input_name = node->inputs();
    params.output_name = node->outputs();

    OpKernelContext* ctx = new OpKernelContext(&params);
    node->op()->Compute(ctx);

    // TODO  delete ctx
  }
  return Status::OK();
}

}
