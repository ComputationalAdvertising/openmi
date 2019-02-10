cmake_minimum_required(VERSION 2.4)

project(openmi-mix)

set(CMAKE_VERBOSE_MAKEFILE on)

set(BASIC_CXXFLAGS "-std=c++11 -g -O3 -Wall -fPIC")
set(BASIC_CPPDEFINES "-DNDEBUG -DHAVE_NETINET_IN_H") 
set(CMAKE_CXX_FLAGS "${BASIC_CXXFLAGS} ${BASIC_CPPDEFINES} -DEIGEN_USE_THREADS")

include_directories(
  $ENV{DEPS_ENV_HOME}/include  
  $ENV{DEPS_ENV_HOME}/include/eigen3
  ${PROJECT_SOURCE_DIR}/../openmi-base
  ${PROJECT_SOURCE_DIR}/../openmi-idl
  ${PROJECT_SOURCE_DIR}/openmi
  ${PROJECT_SOURCE_DIR}/openmi/core/new
  ${PROJECT_SOURCE_DIR}
)

link_directories(
  $ENV{DEPS_ENV_HOME}/lib 
  ${PROJECT_SOURCE_DIR}/../openmi-base/lib
  ${PROJECT_SOURCE_DIR}/../openmi-idl/lib
  ${PROJECT_SOURCE_DIR}/lib
  ${PROJECT_SOURCE_DIR}
  /usr/lib64
)
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
  link_directories($ENV{DEPS_ENV_HOME}/lib64)
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) 

set(STATIC_LINKER_FLAGS "-Wl,-Bstatic ${DEPS_WHOLE_ARCHIVE_LIBRARIES} ${DEPS_STATIC_LIBRARIES}")
set(SHARED_LINKER_FLAGS "-Wl,-Bdynamic ${DEPS_SHARED_LIBRARIES}")
set(DYNAMIC_LINKER "-Wl,--dynamic-linker=${DEPS_DYNAMIC_LINKER}")
set(summary_linker_flags "${STATIC_LINKER_FLAGS} ${SHARED_LINKER_FLAGS} ${DYNAMIC_LINKER}")


add_subdirectory(openmi)
add_subdirectory(unittest) 
