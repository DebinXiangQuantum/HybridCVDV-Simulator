cmake_minimum_required(VERSION 3.18)

# 设置CUDA环境变量
set(ENV{CUDA_HOME} "/usr/lib/nvidia-cuda-toolkit")
set(ENV{LD_LIBRARY_PATH} "$ENV{CUDA_HOME}/lib64:$ENV{LD_LIBRARY_PATH}")

# 设置CUDA架构 (必须在project()之前)
set(CMAKE_CUDA_ARCHITECTURES 60 70 80)

# 指定CUDA主机编译器
set(CMAKE_CUDA_HOST_COMPILER "g++")
set(CUDA_HOST_COMPILER "g++")

project(HybridCVDV-Simulator VERSION 1.5 LANGUAGES CXX)

# 尝试查找CUDA (可选)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found - building with GPU support")
    message(STATUS "CUDA version: ${CUDA_VERSION_STRING}")
    message(STATUS "CUDA toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")

    # 设置CUDA编译选项
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

else()
    message(WARNING "CUDA not found - building CPU-only version")
endif()

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(WARNING "CUDA not found. Building CPU-only version.")
    # 创建一个模拟的CUDA目标用于编译检查
    add_library(cuda_mock INTERFACE)
    target_compile_definitions(cuda_mock INTERFACE CUDA_FOUND=0)
endif()

# 查找Google Test (如果需要的话)，如果未找到则自动下载
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "GTest not found in system, downloading via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://gitee.com/mirrors/googletest.git
        GIT_TAG        v1.14.0
    )
    # 强制使用共享CRT，防止Windows下的链接错误（对Linux无害）
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(googletest)
    
    # FetchContent 提供的目标通常是 gtest_main，这里创建别名以兼容 GTest::gtest_main
    if(NOT TARGET GTest::gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
    
    set(GTest_FOUND TRUE)
endif()

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/reference)

# 源文件列表
set(CORE_SOURCES
    src/core/cv_state_pool.cpp
    src/core/fock_ell_operator.cpp
    src/core/hdd_node.cpp
)

if(CUDA_FOUND)
    set(GPU_SOURCES
        src/gpu/diagonal_gates.cu
        src/gpu/ladder_gates.cu
        src/gpu/single_mode_gates.cu
        src/gpu/two_mode_gates.cu
        src/gpu/hybrid_gates.cu
    )
else()
    set(GPU_SOURCES
        # GPU代码的CPU模拟版本 (如果需要的话)
    )
endif()

set(CPU_SOURCES
    src/cpu/quantum_circuit.cpp
)

set(SCHEDULER_SOURCES
    src/scheduler/batch_scheduler.cpp
    src/scheduler/instruction_fusion.cpp
)

set(MEMORY_SOURCES
    src/memory/memory_pool.cpp
    src/memory/garbage_collector.cpp
)

set(UTILS_SOURCES
    # src/utils/complex_math.cpp
    # src/utils/matrix_operations.cpp
)

set(REFERENCE_SOURCES
    src/reference/reference_gates.cpp
)

# 创建库
add_library(${PROJECT_NAME}_core ${CORE_SOURCES})
if(CUDA_FOUND)
    add_library(${PROJECT_NAME}_gpu ${GPU_SOURCES})
endif()
add_library(${PROJECT_NAME}_cpu ${CPU_SOURCES})
add_library(${PROJECT_NAME}_scheduler ${SCHEDULER_SOURCES})
add_library(${PROJECT_NAME}_memory ${MEMORY_SOURCES})

# 设置目标属性
if(CUDA_FOUND)
    set_target_properties(${PROJECT_NAME}_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# 主库
add_library(${PROJECT_NAME} STATIC
    ${CORE_SOURCES}
    ${GPU_SOURCES}
    ${CPU_SOURCES}
    ${SCHEDULER_SOURCES}
    ${MEMORY_SOURCES}
    ${REFERENCE_SOURCES}
)

# 链接CUDA运行时
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
endif()

# 主可执行文件
add_executable(${PROJECT_NAME}_main src/main.cpp)
target_link_libraries(${PROJECT_NAME}_main ${PROJECT_NAME})
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME}_main ${CUDA_LIBRARIES})
endif()

# 验证程序
add_executable(${PROJECT_NAME}_validation tests/test_validation.cpp)
target_link_libraries(${PROJECT_NAME}_validation ${PROJECT_NAME})

# GPU验证程序
if(CUDA_FOUND)
    add_executable(${PROJECT_NAME}_gpu_validation tests/test_gpu_validation.cpp)
    target_link_libraries(${PROJECT_NAME}_gpu_validation ${PROJECT_NAME} ${CUDA_LIBRARIES})
else()
    # CPU-only版本的GPU验证程序
    add_executable(${PROJECT_NAME}_gpu_validation tests/test_gpu_validation.cpp)
    target_link_libraries(${PROJECT_NAME}_gpu_validation ${PROJECT_NAME})
    target_compile_definitions(${PROJECT_NAME}_gpu_validation PRIVATE CPU_ONLY_GPU_TEST)
endif()

# 示例程序
add_executable(${PROJECT_NAME}_examples examples/basic_usage.cpp)
target_link_libraries(${PROJECT_NAME}_examples ${PROJECT_NAME})
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME}_examples ${CUDA_LIBRARIES})
endif()

# 测试 (如果找到GTest)
if(GTest_FOUND)
    enable_testing()

    set(TEST_SOURCES
        tests/test_cv_state_pool.cpp
        tests/test_fock_ell_operator.cpp
        tests/test_diagonal_gates.cpp
        tests/test_ladder_gates.cpp
        tests/test_gates_implementation.cpp
        tests/test_system.cpp
    )

    add_executable(${PROJECT_NAME}_tests ${TEST_SOURCES})
    target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME} GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(${PROJECT_NAME}_tests)
endif()

# 安装目标
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_main ${PROJECT_NAME}_examples
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})

# 输出配置信息
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA found: ${CUDA_FOUND}")
message(STATUS "CUDA version: ${CUDA_VERSION_STRING}")
message(STATUS "GTest found: ${GTest_FOUND}")
