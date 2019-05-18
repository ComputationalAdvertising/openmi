#include "executor.h"
#include "algorithm.h"
#include "device.h"
#include "device_registry.h"
#include "base/register.h"
#include <set>

using namespace openmi;

bool is_training = true;

namespace openmi {

void FindSinkNodes(Graph& g, std::vector<Node*>& sink_nodes, bool used_back_props = false) {
  // find output nodes 
  std::set<std::string> exist_outputs;
  std::set<std::string> all_node_keys;
  auto it = g.node_mapper().begin();
  while (it != g.node_mapper().end()) {
    Node* n = it->second;
    all_node_keys.insert(n->def().name());
    for (size_t idx = 0; idx < n->inputs().size(); ++idx) {
      exist_outputs.insert(n->inputs()[idx]);
    }
    it++;
  }

  std::set<std::string>::iterator iter;
  for (iter = all_node_keys.begin(); iter != all_node_keys.end(); iter++) {
    if (exist_outputs.find(*iter) == exist_outputs.end()) {
      auto it = g.node_mapper().find(*iter);
      CHECK(it != g.node_mapper().end()) << *iter << " not in node_mapper in graph";
      sink_nodes.push_back(it->second);
    }
  }
}

void Executor::InitSessionState() {
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

    LOG(INFO) << __FUNCTION__ << ". node name: " << kv.second->def().name() << ", is init: " << tensor->IsInitialized();
    if (tensor->IsInitialized()) {
      DLOG(INFO) << "tensor shape: " << tensor->shape().DebugString();
    }
    session_state_.AddTensor(kv.second->def().name(), tensor);
  }
}

Executor::Executor(proto::GraphDef& gdef) {
  Status status = ConvertGraphDefToGraph(&gdef, &g_);

  // not contain reversed node
  FindSinkNodes(g_, g_.sink_nodes());
  
  TopoOrderList(g_.sink_nodes(), g_.forward_topo_nodes(), &g_);

  // TODO CHECK DAG  
  
  // TODO for test
  DLOG(INFO) << "forward topo nodes of graph: ";
  for (int i = 0; i < g_.forward_topo_nodes().size(); ++i) {
    DLOG(INFO) << i << "th. node_name: " << g_.forward_topo_nodes()[i]->def().name();
  }
  
  if (is_training) {
    // Gradients only training phase, not online inference 
    // TODO IF not training, pass it;
    std::vector<Node*>& input_nodes = g_.variable_nodes();
    std::vector<Node*>& output_nodes = g_.sink_nodes();
    std::vector<Node*>& input_gradient_nodes = g_.variable_gradient_nodes();

    int result = gradients_.gradients(output_nodes, input_nodes, input_gradient_nodes, &g_);

    if (result != 0) {
      LOG(FATAL) << "gradients process failed.";
    }

    // re-found sink nodes that contain reversed node
    FindSinkNodes(g_, g_.global_sink_nodes());

    for (int i = 0; i < g_.global_sink_nodes().size(); ++i) {
      DLOG(INFO) << "global sink nodes i[" << i << "] node_name: " << g_.global_sink_nodes().at(i)->def().name();
      if (g_.global_sink_nodes().at(i)->outputs().size() > 0) {
        DLOG(INFO) << "outputs(0): " << g_.global_sink_nodes().at(i)->outputs().at(0);
      }
    }

    TopoOrderList(g_.global_sink_nodes(), g_.global_topo_nodes(), &g_);

    LOG(INFO) << "after topology order list.";
    for (int i = 0; i < g_.global_topo_nodes().size(); ++i) {
      LOG(INFO) << "global topo node. i[" << i << "], node_name: " << g_.global_topo_nodes().at(i)->def().name();
    }
  }

  InitSessionState();

  LOG(INFO) << "Executor init successfully.";
}

Executor::~Executor() {
}

Status Executor::Run() {
  auto& computed_nodes = is_training ? g_.global_topo_nodes() : g_.forward_topo_nodes();
  for (auto& node: computed_nodes) {
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
    params.related_node_name = node->node_info().related_node_name;

    if (node->outputs().size() > 0) {
      LOG(DEBUG) << "output.at(0): " << node->outputs().at(0) << ", node:" << node->def().name();
    }

    LOG(INFO) << "node.name: [" << node->def().name() << "], op: " << node->def().op() << ", related_node: " << node->node_info().related_node_name;
    OpKernelContext* ctx = new OpKernelContext(&params);
    node->op()->Compute(ctx);

    // TODO  delete ctx
  }

  // TODO reversed process 
  return Status::OK();
}

}
