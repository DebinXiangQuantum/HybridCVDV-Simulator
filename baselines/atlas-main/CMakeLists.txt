cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(qsim LANGUAGES C CXX CUDA)
include(ExternalProject)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)
set(QSIM_ROOT ${CMAKE_CURRENT_LIST_DIR})

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LIBEXT ".so")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -UNDEBUG")
add_definitions(-DUSE_RATIONAL)

# Legion to not use static cuda runtime
option(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" ON)

# ##########################################
# custatevec_example build mode
# ##########################################

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build.")
  set_property(
    CACHE
    CMAKE_BUILD_TYPE
    PROPERTY
      STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
else()
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# ##########################################
# custatevec_example dependencies
# ##########################################
set(CUQUANTUM_ROOT "$ENV{CUQUANTUM_ROOT}" CACHE PATH "cuQuantum installation root")
find_path(CUSTATEVEC_INCLUDE_DIR
  NAMES custatevec.h
  HINTS ${CUQUANTUM_ROOT} $ENV{CUQUANTUM_DIR}
  PATH_SUFFIXES include)
find_library(CUSTATEVEC_LIBRARY
  NAMES custatevec
  HINTS ${CUQUANTUM_ROOT} $ENV{CUQUANTUM_DIR}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

if (NOT CUSTATEVEC_LIBRARY OR NOT CUSTATEVEC_INCLUDE_DIR)
    message(FATAL_ERROR "cuStateVec not found; set CUQUANTUM_ROOT, CUSTATEVEC_LIBRARY and CUSTATEVEC_INCLUDE_DIR.")
endif ()

set(CUSTATEVEC_FOUND ON)
set(CUSTATEVEC_LIBRARIES ${CUSTATEVEC_LIBRARY})
set(CUSTATEVEC_INCLUDE_DIRS ${CUSTATEVEC_INCLUDE_DIR})

message(STATUS "CUSTATEVEC include: ${CUSTATEVEC_INCLUDE_DIR}")
message(STATUS "CUSTATEVEC libraries: ${CUSTATEVEC_LIBRARY}")

# ##########################################
# custatevec_example global flags
# ##########################################

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

list(APPEND NVCC_FLAGS
  -std=c++17)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS})

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "CUDA architectures")
endif()

message(STATUS "cuda compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "nvcc flags: ${CUDA_NVCC_FLAGS}")

set(QSIM_INCLUDE_DIRS "")
set(QSIM_EXT_LIBRARIES "")

include(cuda)

include(cuquantum)

if(USE_LEGION)
  include(legion)
endif()

set(NCCL_ROOT "$ENV{NCCL_ROOT}" CACHE PATH "NCCL installation root")
find_path(NCCL_INCLUDE_DIR NAMES nccl.h HINTS ${NCCL_ROOT} PATH_SUFFIXES include)
find_library(NCCL_LIBRARY NAMES nccl HINTS ${NCCL_ROOT} PATH_SUFFIXES lib lib64)
if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR "NCCL not found; set NCCL_ROOT, NCCL_INCLUDE_DIR and NCCL_LIBRARY.")
endif()

find_library(NVML_LIBRARY
  NAMES nvidia-ml
  HINTS /usr/local/cuda /usr/local/cuda/targets/x86_64-linux
  PATH_SUFFIXES lib64/stubs lib/stubs)
if(NOT NVML_LIBRARY)
  message(FATAL_ERROR "NVML link library not found.")
endif()
list(APPEND QSIM_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
list(APPEND QSIM_EXT_LIBRARIES ${NCCL_LIBRARY})

#quartz
add_subdirectory(deps/quartz)

include(mpi)

list(APPEND QSIM_LD_FLAGS
  -lrt
  -ldl
  -rdynamic)
link_libraries(${QSIM_LD_FLAGS})

list(APPEND QSIM_INCLUDE_DIRS
  ${QSIM_ROOT}/include)

file(GLOB_RECURSE QSIM_HDR
  LIST_DIRECTORIES False
  ${QSIM_ROOT}/include/*.h)

if(USE_LEGION)
  file(GLOB_RECURSE QSIM_SRC
    LIST_DIRECTORIES False
    ${QSIM_ROOT}/src/*.cc)

  file(GLOB_RECURSE QSIM_GPU_SRC
    LIST_DIRECTORIES False
    ${QSIM_ROOT}/src/*.cu)
else()
  file(GLOB_RECURSE QSIM_SRC
    LIST_DIRECTORIES False
    ${QSIM_ROOT}/src/base/*.cc)

  file(GLOB_RECURSE QSIM_GPU_SRC
    LIST_DIRECTORIES False
    ${QSIM_ROOT}/src/base/*.cu)
endif()

# 添加性能监控源文件
list(APPEND QSIM_SRC ${QSIM_ROOT}/src/base/performance_monitor.cc)
list(APPEND QSIM_SRC ${QSIM_ROOT}/src/base/performance_monitor_global.cc)


cuda_add_library(quansim SHARED ${QSIM_GPU_SRC} ${QSIM_SRC})
set_property(TARGET quansim PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(quansim PUBLIC ${QSIM_INCLUDE_DIRS})
# TODO: add option to choose simulator backend
if(USE_LEGION)
  target_link_libraries(quansim ${LEGION_LIBRARY} quartz_runtime ${NVML_LIBRARY})
else()
  target_link_libraries(quansim quartz_runtime ${NVML_LIBRARY})
endif()

set(INCLUDE_DEST "include")
set(LIB_DEST "lib")
install(FILES ${QSIM_HDR} DESTINATION ${INCLUDE_DEST})
install(TARGETS quansim DESTINATION ${LIB_DEST})

if(USE_LEGION)
  add_subdirectory(examples/legion-based)
else()
  add_subdirectory(examples/mpi-based)
endif()

