#!/bin/bash 

set -o pipefail
set -o errexit

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
PROJECT_DIR=$SCRIPT_DIR

deps_base_path=$HOME/.openmit_deps

while getopts "c:" opt; do
  case "$opt" in 
    c) deps_base_path=$OPTARG;;
    \?) echo "Invalid option: -$OPTARG";;
  esac
done 

export DEPS_ENV_HOME=$deps_base_path

mkdir -p $PROJECT_DIR/build || echo "$PROJECT_DIR/build exists!"
cd $PROJECT_DIR/build

dep_whole_archive_libraries="-lopenmit_core"
dep_static_libraries="-lopenmi_idl -lopenmi_base \
-lprotobuf -lprotoc -lprotobuf-lite -lgmock -lgtest -lgtest_main"
dep_dynamic_libraries="-lpthread -lgflags"

dynamic_linker=""
os_name=`uname | tr "A-Z" "a-z"` 
if [[ "$os_name" == "linux" ]]; then
  dynamic_linker="/lib64/ld-lsb-x86-64.so"
fi

#build_type="Debug"
build_type="Release"

cmake $PROJECT_DIR \
-DCMAKE_C_COMPILER=`which gcc` \
-DCMAKE_CXX_COMPILER=`which g++` \
-DCMAKE_BUILD_TYPE=$build_type \
-DDEPS_WHOLE_ARCHIVE_LIBRARIES="$dep_whole_archive_libraries" \
-DDEPS_STATIC_LIBRARIES="$dep_static_libraries" \
-DDEPS_SHARED_LIBRARIES="$dep_dynamic_libraries" \
-DDEPS_DYNAMIC_LINKER="$dynamic_linker"

#make
make -j8

echo "======== ${BASH_SOURCE[0]} ========"
