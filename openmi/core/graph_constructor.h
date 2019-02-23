#ifndef OPENMI_CORE_GRAPH_GRAPH_CONSTRUCTOR_H_
#define OPENMI_CORE_GRAPH_GRAPH_CONSTRUCTOR_H_ 

#include "status.h"
#include "graph.h"

namespace openmi {

extern Status ConvertGraphDefToGraph(GraphDef* gdef, Graph* g);

} // namespace openmi
#endif // OPENMI_CORE_GRAPH_GRAPH_CONSTRUCTOR_H_ 
