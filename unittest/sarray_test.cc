#include <stdio.h>
#include <string>
#include <unordered_map>
#include <memory>
#include "sarray.h"
#include "openmi/idl/proto/engine.pb.h"
using namespace openmi::proto::internal;

using namespace mit;

typedef std::shared_ptr<ColumnWeightSchema> ColumnWeightSchemaPtr;

/*
int main(int argc, char** argv) {
  // engine: 根据模型参数（s）与参数定义（schema） 找到对应的参数

  // ps: 根据图梯度（grad）和模型参数定义（schema）拼模型梯度，push给ps

  return 0;
}
*/

/*!
 * 模拟pull操作后，根据column参数（s）和column schema获取参数，用于fill图column node
 */
void GetColumnWeightBySchema(ColumnWeightSchemaPtr& cwsp, const SArray<float>& s, const int index) {
  WeightOffset weight_offset = cwsp->weight_offset(index);
  SArray<float> weight = s.segment(weight_offset.weight_offset(), weight_offset.weight_offset() + weight_offset.weight_size());
  printf("weight: %s\n", DebugStr(weight.data(), weight_offset.weight_size()).c_str());
}

void SetColumnWeightGrad() {

}

int main(int argc, char** argv) {
  ColumnWeightSchemaPtr cwsp = std::make_shared<ColumnWeightSchema>();
  cwsp->set_column_id(1);
  WeightOffset* wo1 = cwsp->add_weight_offset();
  wo1->set_weight_offset(0);
  wo1->set_weight_size(1);
  cwsp->set_total_weight_size(cwsp->total_weight_size() + 1);

  WeightOffset* wo2 = cwsp->add_weight_offset();
  wo2->set_weight_offset(1);
  wo2->set_weight_size(8);
  cwsp->set_total_weight_size(cwsp->total_weight_size() + 8);

  printf("column weight schema:\n%s\n", cwsp->DebugString().c_str());
  SArray<float> s;
  s.resize(cwsp->total_weight_size(), 0.1);

  GetColumnWeightBySchema(cwsp, s, 0);
  GetColumnWeightBySchema(cwsp, s, 1);

  ModelKVPairs* kv_pairs = new ModelKVPairs();
  ValList* vals1 = kv_pairs->add_vals();
  vals1->add_val(0.1);
  vals1->add_val(0.11);
  vals1->add_val(0.111);

  ValList* vals2 = kv_pairs->add_vals();
  vals2->add_val(0.2);
  vals2->add_val(0.22);

  const float* vals = kv_pairs->vals(0).val().data();
  int size = kv_pairs->vals(0).val_size();
  printf("size:%d, 0:%f, 1:%f, 2:%f\n", size, *vals, *(vals+1), *(vals+2));
  printf("vals(0):\n%s\n", kv_pairs->vals(0).DebugString().c_str());

  return 0;
}