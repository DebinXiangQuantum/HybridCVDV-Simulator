cmake_minimum_required(VERSION 3.10)
project(BosonicOperators VERSION 1.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加CUDA支持
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    enable_language(CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++17)
    message(STATUS "CUDA found: ${CUDA_VERSION_STRING}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
else()
    message(STATUS "CUDA not found, only CPU version will be built")
endif()

# 设置包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# CPU库
add_library(bosonic_cpu STATIC
    cpu/operators.cpp
)

# GPU库 (如果CUDA可用)
if(CUDA_FOUND)
    add_library(bosonic_gpu STATIC
        gpu/operators.cu
    )
    target_include_directories(bosonic_gpu PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(bosonic_gpu ${CUDA_LIBRARIES} cublas cusparse)
    
    # 设置 CUDA 架构
    set_target_properties(bosonic_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;86;89"
    )
endif()

# 主程序
add_executable(bosonic_test
    main.cpp
)

target_link_libraries(bosonic_test bosonic_cpu)

if(CUDA_FOUND)
    target_link_libraries(bosonic_test bosonic_gpu)
endif()

# GPU 简单测试程序
if(CUDA_FOUND)
    add_executable(test_gpu_simple
        test_gpu_simple.cpp
    )
    target_link_libraries(test_gpu_simple bosonic_gpu ${CUDA_LIBRARIES})
    
    # GPU 加速验证程序
    add_executable(verify_gpu_acceleration
        verify_gpu_acceleration.cpp
    )
    target_link_libraries(verify_gpu_acceleration bosonic_gpu ${CUDA_LIBRARIES})
endif()

# 设置输出目录
set_target_properties(bosonic_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# 安装配置
install(TARGETS bosonic_test
    RUNTIME DESTINATION bin
)

install(TARGETS bosonic_cpu
    ARCHIVE DESTINATION lib
)

if(CUDA_FOUND)
    install(TARGETS bosonic_gpu
        ARCHIVE DESTINATION lib
    )
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cpu
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

if(CUDA_FOUND)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gpu
        DESTINATION include
        FILES_MATCHING PATTERN "*.cuh"
    )
endif()