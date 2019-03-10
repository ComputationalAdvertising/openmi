#include "attr_value.h"

namespace openmi {

void AttrValue::FromProto(const proto::AttrValue& attr) {
  switch(attr.value_case()) {
    case proto::AttrValue::kS: 
      {
        this->attr_type = kString;
        this->s = attr.s();
        break;
      }
    case proto::AttrValue::kI: 
      {
        this->attr_type = kInt;
        this->i = attr.i();
        break;
      }
    case proto::AttrValue::kF: 
      {
        this->attr_type = kFloat;
        this->b = attr.f();
        break;
      }
    case proto::AttrValue::kB: 
      {
        this->attr_type = kBool;
        this->b = attr.b();
        break;
      }
    case proto::AttrValue::kType: 
      {
        this->attr_type = kType;
        this->type = attr.type();
        break;
      }
    case proto::AttrValue::kShape: 
      {
        this->attr_type = kShape;
        this->shape = TensorShape(attr.shape());
        break;
      }
    case proto::AttrValue::kTensor: 
      {
        this->attr_type = kTensor;
        //this->tensor = Tensor(attr.tensor());
        break;
      }
    case proto::AttrValue::VALUE_NOT_SET:
      {
        this->attr_type = kNone;
        break;
      }
  }
}

proto::AttrValue ToProto() {
  proto::AttrValue rt;
  // TODO 
  return rt;
}

}
