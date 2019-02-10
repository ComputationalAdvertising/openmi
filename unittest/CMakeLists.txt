cmake_minimum_required(VERSION 2.4) 

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(PREFIX_LINKER_FLAGS "-static -Wl,--whole-archive ${DEPS_WHOLE_ARCHIVE_LIBRARIES} -Wl,--no-whole-archive -Wl,--eh-frame-hdr")

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(summary_linker_flags "${STATIC_LINKER_FLAGS} ${SHARED_LINKER_FLAGS} ${DYNAMIC_LINKER}")
else()
  set(summary_linker_flags "${DEPS_WHOLE_ARCHIVE_LIBRARIES} ${DEPS_STATIC_LIBRARIES} ${DEPS_SHARED_LIBRARIES}")
endif()

function (add_program arg)
  list(LENGTH ARGV  argv_len)
  if (NOT (${argv_len} STREQUAL "1")) 
    message(FATAL_ERROR "argment number should be equals to 1")
  endif()

  set(prog_name ${ARGV0})
  message(STATUS "exec_name: ${prog_name}, src_name: ${prog_name}.cc")
  add_executable(${prog_name} "${prog_name}.cc")
  target_link_libraries(${prog_name} ${summary_linker_flags})
endfunction()


#add_program("op_test")
#add_program("autodiff_test")
#add_program("variable_test")
#add_program("tensor_shape_test")
#add_program("tensor_test") 

#add_program("core_framework_executor_test")
#add_program("core_framework_gradients_test")
#add_program("core_framework_node_manager_test")
#add_program("core_framework_op_test")
#add_program("core_framework_tensor_buffer_test")
#add_program("core_framework_tensor_shape_test")
add_program("core_framework_tensor_test")
#add_program("core_framework_types_test")
