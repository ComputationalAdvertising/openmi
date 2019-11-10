#ifndef OPENMI_CORE_ENGINE_NN_STRUCTURE_H_
#define OPENMI_CORE_ENGINE_NN_STRUCTURE_H_

#include <stdlib.h>
#include <string>

namespace openmi {
namespace engine {

struct NNEntry {
  uint64_t fid;
  int node_id;
  NNEntry(uint64_t _fid, int _node_id) : fid(_fid), node_id(_node_id) {}
}; // struct NNEntry

struct NNVariableInfo {
  std::string node_name;
  std::string reversed_node_name;
  int node_id;
  explicit NNVariableInfo(std::string _name, std::string _reversed_name, int id) 
    : node_name(_name), reversed_node_name(_reversed_name), node_id(id) {}
}; // struct NNNodeInfo

} // namespace engine
} // namespace openmi
#endif // OPENMI_CORE_ENGINE_NN_STRUCTURE_H_