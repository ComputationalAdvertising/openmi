#ifndef OPENMI_CORE_ENGINE_NN_STRUCTURE_H_
#define OPENMI_CORE_ENGINE_NN_STRUCTURE_H_

#include <stdlib.h>

namespace openmi {
namespace engine {

struct NNEntry {
  NNEntry(uint64_t _fid, int _node_id): fid(_fid), node_id(_node_id) {}
  uint64_t fid;
  int node_id;
}; // struct NNEntry

}
}
#endif // OPENMI_CORE_ENGINE_NN_STRUCTURE_H_