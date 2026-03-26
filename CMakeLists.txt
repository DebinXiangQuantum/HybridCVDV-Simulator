cmake_minimum_required(VERSION 3.18)

# 优先尊重外部 CUDA 配置，未显式提供时再回退到常见安装路径。
if(NOT DEFINED ENV{CUDA_HOME} OR "$ENV{CUDA_HOME}" STREQUAL "")
    if(EXISTS "/usr/local/cuda")
        set(ENV{CUDA_HOME} "/usr/local/cuda")
    elseif(EXISTS "/usr/lib/nvidia-cuda-toolkit")
        set(ENV{CUDA_HOME} "/usr/lib/nvidia-cuda-toolkit")
    endif()
endif()

if((NOT DEFINED CUDAToolkit_ROOT OR "${CUDAToolkit_ROOT}" STREQUAL "") AND DEFINED ENV{CUDA_HOME} AND NOT "$ENV{CUDA_HOME}" STREQUAL "")
    set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}")
endif()

if((NOT DEFINED CMAKE_CUDA_COMPILER OR "${CMAKE_CUDA_COMPILER}" STREQUAL "") AND DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc")
endif()

if(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/lib64")
    set(ENV{LD_LIBRARY_PATH} "$ENV{CUDA_HOME}/lib64:$ENV{LD_LIBRARY_PATH}")
endif()

# 设置CUDA架构 (必须在project()之前)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    if(DEFINED ENV{HYBRIDCVDV_CUDA_ARCHITECTURES} AND NOT "$ENV{HYBRIDCVDV_CUDA_ARCHITECTURES}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "$ENV{HYBRIDCVDV_CUDA_ARCHITECTURES}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES 60 70 80)
    endif()
endif()

# 指定CUDA主机编译器 (默认gcc-9, 可通过命令行覆盖)
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    if(EXISTS "/usr/bin/gcc-9")
        set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-9")
        set(CUDA_HOST_COMPILER "/usr/bin/gcc-9")
    else()
        # 回退到系统默认gcc
        find_program(_GCC_PATH gcc)
        if(_GCC_PATH)
            set(CMAKE_CUDA_HOST_COMPILER "${_GCC_PATH}")
            set(CUDA_HOST_COMPILER "${_GCC_PATH}")
        endif()
    endif()
endif()

project(HybridCVDV-Simulator VERSION 1.5 LANGUAGES CXX)

option(HYBRIDCVDV_BUILD_TESTS "Build the GoogleTest-based test suite" OFF)

# 尝试查找CUDA (可选)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA found - building with GPU support")
    message(STATUS "CUDA version: ${CUDA_VERSION_STRING}")
    message(STATUS "CUDA toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")

    # 设置CUDA编译选项
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
    endif()

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

# Google Test - 仅在显式启用测试时下载
if(HYBRIDCVDV_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://gitee.com/mirrors/googletest.git
        GIT_TAG v1.14.0
    )
    # 防止googletest项目覆盖我们的编译设置
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/reference)
if(CUDA_FOUND)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()
# 源文件列表
set(CORE_SOURCES
    src/core/cv_state_pool.cpp
    src/core/fock_ell_operator.cpp
    src/core/hdd_node.cpp
    src/core/gaussian_state.cpp
    src/core/gaussian_mixture.cpp
    src/core/symplectic_math.cpp
    src/core/gaussian_circuit.cpp
    src/core/gaussian_projection.cpp
)

if(CUDA_FOUND)
    set(GPU_SOURCES
        src/gpu/diagonal_gates.cu
        src/gpu/ladder_gates.cu
        src/gpu/single_mode_gates.cu
        src/gpu/single_mode_xzpv_gates.cu
        src/gpu/two_mode_gates.cu
        src/gpu/two_mode_special_gates.cu
        src/gpu/hybrid_gates.cu
        src/gpu/hybrid_interaction_gates.cu
        src/gpu/squeezing_gate.cu
        src/gpu/qubit_gates.cu
        src/gpu/snap_gates.cu
        src/gpu/gaussian_kernels.cu
        src/gpu/gpu_context.cu
        src/gpu/stream_pool.cu
        src/gpu/peer_transfer.cu
    )
else()
    set(GPU_SOURCES
        # GPU代码的CPU模拟版本 (如果需要的话)
    )
endif()

set(CPU_SOURCES
    src/cpu/quantum_circuit.cpp
)

if(CMAKE_BUILD_TYPE STREQUAL "Release" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        src/cpu/quantum_circuit.cpp
        PROPERTIES COMPILE_OPTIONS "-O1"
    )
endif()

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
    src/reference/squeezing_matrix.cpp
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
if(CUDA_FOUND)
    add_library(${PROJECT_NAME} STATIC
        ${CORE_SOURCES}
        ${GPU_SOURCES}
        ${CPU_SOURCES}
        ${SCHEDULER_SOURCES}
        ${MEMORY_SOURCES}
        ${REFERENCE_SOURCES}
    )
else()
    add_library(${PROJECT_NAME} STATIC
        ${CORE_SOURCES}
        ${CPU_SOURCES}
        ${SCHEDULER_SOURCES}
        ${MEMORY_SOURCES}
        ${REFERENCE_SOURCES}
    )
endif()

# 链接CUDA运行时
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} CUDA::cublas)
    if(TARGET CUDA::nvToolsExt)
        target_link_libraries(${PROJECT_NAME} CUDA::nvToolsExt)
        target_compile_definitions(${PROJECT_NAME} PRIVATE HYBRIDCVDV_HAS_NVTX=1)
    else()
        target_compile_definitions(${PROJECT_NAME} PRIVATE HYBRIDCVDV_HAS_NVTX=0)
    endif()
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE HYBRIDCVDV_HAS_NVTX=0)
endif()

# 主可执行文件
add_executable(${PROJECT_NAME}_main src/main.cpp)
target_link_libraries(${PROJECT_NAME}_main ${PROJECT_NAME})
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME}_main ${CUDA_LIBRARIES} CUDA::cublas)
endif()

# 验证程序已包含在主测试套件中
# GPU验证程序已包含在主测试套件中

# 示例程序
add_executable(${PROJECT_NAME}_examples examples/basic_usage.cpp)
target_link_libraries(${PROJECT_NAME}_examples ${PROJECT_NAME})
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME}_examples ${CUDA_LIBRARIES} CUDA::cublas)
endif()

# 单卡实验程序
if(CUDA_FOUND)
    add_executable(hybridcvdv_single_gpu_experiments
        experiments/cpp/single_gpu_experiments.cpp
    )
    target_link_libraries(hybridcvdv_single_gpu_experiments ${PROJECT_NAME} ${CUDA_LIBRARIES} CUDA::cublas)
    if(CMAKE_BUILD_TYPE STREQUAL "Release" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # This benchmark driver is a very large translation unit and can be killed during
        # optimization on shared remote hosts. Lowering only the driver TU keeps the core
        # simulator library in Release mode while making the benchmark binary build reliably.
        target_compile_options(hybridcvdv_single_gpu_experiments PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O1>)
    endif()
endif()

# Profiling Test
add_executable(${PROJECT_NAME}_profiling_test tests/analyze/profiling_test.cpp)
target_link_libraries(${PROJECT_NAME}_profiling_test ${PROJECT_NAME})
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME}_profiling_test ${CUDA_LIBRARIES})
endif()

add_executable(${PROJECT_NAME}_gaussian_order_benchmark tests/analyze/gaussian_order_benchmark.cpp)
target_link_libraries(${PROJECT_NAME}_gaussian_order_benchmark ${PROJECT_NAME})
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME}_gaussian_order_benchmark ${CUDA_LIBRARIES} CUDA::cublas)
endif()

add_executable(${PROJECT_NAME}_gaussian_mixture_benchmark tests/analyze/gaussian_mixture_benchmark.cpp)
target_link_libraries(${PROJECT_NAME}_gaussian_mixture_benchmark ${PROJECT_NAME})
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME}_gaussian_mixture_benchmark ${CUDA_LIBRARIES} CUDA::cublas)
endif()

add_executable(${PROJECT_NAME}_random_circuit_benchmark tests/analyze/random_circuit_benchmark.cpp)
target_link_libraries(${PROJECT_NAME}_random_circuit_benchmark ${PROJECT_NAME})
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME}_random_circuit_benchmark ${CUDA_LIBRARIES} CUDA::cublas)
endif()

add_executable(${PROJECT_NAME}_two_mode_tensor_benchmark tests/analyze/two_mode_tensor_benchmark.cpp)
target_link_libraries(${PROJECT_NAME}_two_mode_tensor_benchmark ${PROJECT_NAME})
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME}_two_mode_tensor_benchmark ${CUDA_LIBRARIES} CUDA::cublas)
endif()

add_executable(${PROJECT_NAME}_cuda_graph_benchmark tests/analyze/cuda_graph_benchmark.cpp)
target_link_libraries(${PROJECT_NAME}_cuda_graph_benchmark ${PROJECT_NAME})
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME}_cuda_graph_benchmark ${CUDA_LIBRARIES} CUDA::cublas)
endif()

if(CUDA_FOUND)
    add_executable(${PROJECT_NAME}_gpu_precision_report tests/analyze/gpu_precision_report.cpp)
    target_link_libraries(${PROJECT_NAME}_gpu_precision_report ${PROJECT_NAME} ${CUDA_LIBRARIES} CUDA::cublas)
endif()

# 测试 (使用自动下载的GoogleTest)
if(HYBRIDCVDV_BUILD_TESTS)
    enable_testing()

    set(TEST_SOURCES
        tests/test_cv_state_pool.cpp
        tests/test_fock_ell_operator.cpp
        tests/test_all_gates.cpp
        tests/test_diagonal_gates.cpp
        tests/test_ladder_gates.cpp
        tests/test_hdd_node.cpp
        tests/test_system.cpp
        tests/test_snap_gates.cpp
        tests/test_sf_two_mode_gates.cpp
        tests/test_sf_single_mode_gates.cpp
        tests/test_number_operator.cpp
        tests/test_qubit_gates.cpp
        tests/test_validation.cpp
        tests/test_gpu_validation.cpp
        tests/test_gaussian_symplectic.cpp
        tests/test_gaussian_mixture.cpp
    )

    add_executable(${PROJECT_NAME}_tests ${TEST_SOURCES})
    target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME} GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(${PROJECT_NAME}_tests)

    # Multi-GPU unit tests
    if(CUDA_FOUND)
        set(MULTI_GPU_TEST_SOURCES
            src/test/test_gpu_context.cpp
            src/test/test_stream_pool.cpp
            src/test/test_peer_transfer.cpp
            src/test/test_multi_gpu_state.cpp
            src/test/test_multi_gpu_scheduler.cpp
            src/test/test_multi_gpu_integration.cpp
        )

        add_executable(${PROJECT_NAME}_multi_gpu_tests ${MULTI_GPU_TEST_SOURCES})
        target_link_libraries(${PROJECT_NAME}_multi_gpu_tests
            ${PROJECT_NAME} GTest::gtest_main ${CUDA_LIBRARIES} CUDA::cublas)
        gtest_discover_tests(${PROJECT_NAME}_multi_gpu_tests)
    endif()
endif()

# 安装目标
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_main ${PROJECT_NAME}_examples
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

if(CUDA_FOUND)
    install(TARGETS hybridcvdv_single_gpu_experiments
        RUNTIME DESTINATION bin
    )
endif()

install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})

# 输出配置信息
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA found: ${CUDA_FOUND}")
message(STATUS "CUDA version: ${CUDA_VERSION_STRING}")
message(STATUS "Build tests: ${HYBRIDCVDV_BUILD_TESTS}")
if(HYBRIDCVDV_BUILD_TESTS)
    message(STATUS "GTest: Auto-downloaded and configured")
endif()
