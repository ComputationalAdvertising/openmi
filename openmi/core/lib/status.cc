#include "core/lib/status.h"
#include <assert.h>
#include <stdio.h>

using namespace openmi;

namespace openmi {

Status::Status(Code code, std::string msg) {
  assert(code != openmi::OK);
  state_ = std::unique_ptr<State>(new State);
  state_->code = code;
  state_->msg = msg;
}

void Status::Update(const Status& new_status) {
  if (ok()) {
    *this = new_status;
  }
}

void Status::SlowCopyFrom(const State* s) {
  if (s == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*s));
  }
}

std::string Status::ToString() const {
  if (state_ == nullptr) {
    return "OK";
  } else {
    return "OutOfRange";
    // TODO 
  }
  return "Status::ToString";
}

std::ostream& operator<<(std::ostream& os, const Status& s) {
  os << s.ToString();
  return os;
}

}
