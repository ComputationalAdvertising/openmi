cmake_minimum_required(VERSION 2.4)

aux_source_directory(core/common_runtime srcs_core_common_runtime)
aux_source_directory(core/framework srcs_core_framework)
aux_source_directory(core/graph srcs_core_graph)
aux_source_directory(core/lib srcs_core_lib)
aux_source_directory(core/ops srcs_core_ops)
aux_source_directory(core/test srcs_core_test)

set(srcs ${srcs_core_test})
#set(srcs ${srcs_core_framework})
#list(APPEND srcs ${srcs_core_graph})
#list(APPEND srcs ${srcs_core_lib})
#list(APPEND srcs ${srcs_core_ops})
#list(APPEND srcs ${srcs_core_common_runtime})

set(libname openmit_core)
add_library(${libname} STATIC ${srcs})
target_link_libraries(${libname} "${summary_linker_flags}")
