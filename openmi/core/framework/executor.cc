#include "executor.h"
#include "algorithm.h"
#include "device.h"
#include "graph_utils.h"
#include "device_registry.h"
#include "base/register.h"
#include <set>

using namespace openmi;

bool is_training = true;

namespace openmi {

Executor::Executor(proto::GraphDef& gdef) 
  : g_(nullptr),
    gradients_(nullptr),
    session_state_(nullptr) {
  Init(gdef);
}

Executor::~Executor() {
  Destroy();
}

void Executor::Init(proto::GraphDef& gdef) {
  g_ = std::make_shared<Graph>();
  Status status = ConvertGraphDefToGraph(&gdef, g_.get());
  // only forward node
  FindSourceNodes(g_.get(), g_->source_nodes());
  FindSinkNodes(g_.get(), g_->sink_nodes());
  TopoOrderList(g_->sink_nodes(), g_->forward_topo_nodes(), g_.get());

  // TODO CHECK DAG  
  
  // Gradients that only training phase, not contain inference phase
  if (is_training) {
    std::vector<Node*>& input_variable_nodes = g_->variable_nodes();
    std::vector<Node*>& output_nodes = g_->sink_nodes();

    gradients_ = std::make_shared<Gradients>();
    session_state_ = std::make_shared<SessionState>();
    int result = gradients_->gradients(output_nodes, input_variable_nodes, g_.get(), session_state_.get());
    CHECK(result == 0) << "gradients process failed when training task.";

    // refound sink nodes that contain reversed node
    FindSinkNodes(g_.get(), g_->global_sink_nodes());
    TopoOrderList(g_->global_sink_nodes(), g_->global_topo_nodes(), g_.get());
    
    DebugGraphNodes(g_.get());
  }

  InitSessionState();
  InitComputeOp();

  LOG(INFO) << "global nodes size[" << g_->global_topo_nodes().size() 
            << "], source nodes size[" << g_->source_nodes().size()
            << "], sink nodes size[" << g_->sink_nodes().size()
            << "], forward nodes size[" << g_->forward_topo_nodes().size()
            << "], backward nodes size[" << g_->reversed_nodes().size() << "]";
  LOG(INFO) << "Executor init successfully.";
}

void Executor::Destroy() {
  auto it = node_kernel_context_mapper_.begin();
  for (; it != node_kernel_context_mapper_.end(); ++it) {
    it->second->Destroy();
    delete it->second;
    it->second = nullptr;
  }
  node_kernel_context_mapper_.clear();

  g_->Destroy();
}

void Executor::InitSessionState() {
  for (auto& kv: g_->node_mapper()) {
    DataType type = DT_FLOAT;
    auto it = kv.second->attrs().find("type");
    if (it != kv.second->attrs().end()) {
      CHECK(it->second.attr_type == ::openmi::AttrValue::kType);
      type = it->second.type;
    }

    std::string node_name = kv.second->def().name();
    Tensor* tensor;
    it = kv.second->attrs().find("shape");
    if (it != kv.second->attrs().end() && 
        it->second.attr_type == ::openmi::AttrValue::kShape) {
      tensor = new Tensor(type, it->second.shape);
      DLOG(INFO) << node_name << ", shape:" << it->second.shape.DebugString();
    } else {
      tensor = new Tensor(type);
    }

    session_state_->AddTensor(node_name, tensor);

    DLOG(INFO) << __FUNCTION__ << "node name: " << node_name << " init tensor done.";
  }
}

void Executor::InitComputeOp() {
  compute_nodes_ = is_training ? g_->global_topo_nodes() : g_->forward_topo_nodes();
  for (auto& node: compute_nodes_) {
    OpKernelConstruction okc(node->def().name(), node->attrs());
    node->op()->Initialize(&okc);

    OpKernelContext::Params* params_ptr = new OpKernelContext::Params();
    auto device = node->def().device();
    const DeviceFactory* device_factory = openmi::Register<DeviceFactory>::Find(device);
    CHECK(device_factory != nullptr) << "device not exists. device:" << device;
    params_ptr->device = device_factory->func();
    params_ptr->session_state = session_state_.get();
    params_ptr->op_kernel = node->op();
    params_ptr->node_def = &node->def();
    params_ptr->input_name = node->inputs();
    params_ptr->output_name = node->outputs();
    params_ptr->related_node_name = node->node_info().related_node_name;

    DLOG(INFO) << "node def:\n" << params_ptr->node_def->DebugString();
    OpKernelContext* context = new OpKernelContext(params_ptr);
    node_kernel_context_mapper_.insert({node->def().name(), context});
  }
}

SessionState* Executor::GetSessionState() {
  CHECK(session_state_ != nullptr);
  return session_state_.get();
}

Graph* Executor::GetGraph() {
  CHECK(g_ != nullptr);
  return g_.get();
}

Status Executor::Run() {
  for (auto& node: compute_nodes_) {
    /*
    OpKernelConstruction okc(node->def().name(), node->attrs());
    node->op()->Initialize(&okc);
    
    OpKernelContext::Params params;
    
    auto device_name = node->def().device();
    const DeviceFactory* device_factory = openmi::Register<DeviceFactory>::Find(device_name);
    CHECK(device_factory != nullptr) << "deivce not exist. device:" << device_name;
    Device* device = device_factory->func();
    params.device = device;
    
    params.session_state = &session_state_;
    params.op_kernel = node->op();
    params.node_def = &node->def();
    params.input_name = node->inputs();
    params.output_name = node->outputs();
    params.related_node_name = node->node_info().related_node_name;

    if (node->outputs().size() > 0) {
      DLOG(INFO) << "output.at(0): " << node->outputs().at(0) << ", node:" << node->def().name();
    }

    DLOG(INFO) << "node.name: [" << node->def().name() << "], op: " << node->def().op();
    // 思考：是否每次都需要new OpKernelContext呢？？？
    OpKernelContext* ctx = new OpKernelContext(&params);
    // 设计一个 map<node, OpKernelContext>，没必要每次都初始化OpKernelContext
    */
    DLOG(INFO) << node->def().DebugString();
    auto it = node_kernel_context_mapper_.find(node->def().name());
    CHECK(it != node_kernel_context_mapper_.end()) 
      << "OpKernelContext not exists. node:" << node->def().DebugString();
    OpKernelContext* context = it->second;
    CHECK(context != nullptr);
    DLOG(INFO) << "node:" << context->name() << ", relate node:" << context->related_node_name();
    node->op()->Compute(context);
  }

  // TODO reversed process 
  return Status::OK();
}

}
