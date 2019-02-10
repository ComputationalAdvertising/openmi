cmake_minimum_required(VERSION 2.4)

aux_source_directory(core/lib srcs_core_lib)

set(srcs ${srcs_core_lib})

set(libname openmi_core)
add_library(${libname} STATIC ${srcs})
target_link_libraries(${libname} "${summary_linker_flags}")
