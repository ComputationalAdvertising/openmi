#include "core/framework/op_factory.h"

namespace openmi {

OPENMI_REGISTER_ENABLE(OpFactory);

// Notice: This macro must be used in the same tag_name as OPENMI_REGISTER_FILE_TAG
OPENMI_REGISTER_LINK_TAG(add_op);
OPENMI_REGISTER_LINK_TAG(multiply_op);
OPENMI_REGISTER_LINK_TAG(oneslike_op);
OPENMI_REGISTER_LINK_TAG(placeholder_op);
OPENMI_REGISTER_LINK_TAG(zeroslike_op);
// TODO  ....

}
