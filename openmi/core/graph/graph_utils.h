#ifndef OPENMI_CORE_FRAMEWORK_GRAPH_UTILS_H_
#define OPENMI_CORE_FRAMEWORK_GRAPH_UTILS_H_ 

#include "graph.h"

namespace openmi {

extern Node* CreateGradNode(proto::NodeDef& ndef, Graph& g, const std::string& related_node_name, NodeClass nc = NC_OP, NodeScope ns = NS_REVERSE);

extern Node* CreateGradNode(const std::string& node_name, const std::string& op, Graph& g, const std::string& related_node_name, NodeClass nc = NC_OP, NodeScope ns = NS_REVERSE);

} // namespace openmi 
#endif // OPENMI_CORE_FRAMEWORK_GRAPH_UTILS_H_ 
