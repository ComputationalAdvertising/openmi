cmake_minimum_required(VERSION 2.4)

aux_source_directory(core/framework srcs_core_framework)
aux_source_directory(core/graph srcs_core_graph)
aux_source_directory(core/ops srcs_core_ops)
aux_source_directory(core/engine srcs_core_engine)

set(srcs ${srcs_core_framework})
list(APPEND srcs ${srcs_core_graph})
list(APPEND srcs ${srcs_core_ops})
list(APPEND srcs ${srcs_core_engine})

set(libname openmi_core)

#add_library(${libname} SHARED ${srcs})
#target_link_libraries(${libname} ${link_libs})

add_library(${libname}_static STATIC ${srcs})
target_link_libraries(${libname}_static ${summary_linker_flags})
set_target_properties(${libname}_static PROPERTIES OUTPUT_NAME ${libname})
