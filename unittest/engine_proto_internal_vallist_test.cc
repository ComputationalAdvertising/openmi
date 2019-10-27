#include "openmi/idl/proto/communication.pb.h"
#include "base/logging.h"
using namespace openmi;

int main(int argc, char** argv) {
  
  proto::comm::ValueList val_list;
  val_list.mutable_val()->Resize(9, 0);
  val_list.add_val(0.1);
  // val_list.set_val(2, 0.2);
  // val_list.set_val(9, 0.9);
  LOG(INFO) << val_list.DebugString();

  proto::comm::CommData model_grad_data;
  proto::comm::ValueList* grad_vals = model_grad_data.add_vals();
  grad_vals->MergeFrom(val_list);
  model_grad_data.add_keys(9);
  LOG(INFO) << "model grad data:\n" << model_grad_data.DebugString();
  return 0;
}