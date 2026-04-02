cmake_minimum_required(VERSION 3.8)
project(bosonic_circuits)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

set(CUDA_ARCHITECTURES 61 70 75 80 86)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(src)
include_directories(.)
include_directories(include)

set(COMMON_SOURCES
    src/core/quantum_state.cpp
    src/core/circuit.cpp
)

set(CUDA_SOURCES
    src/gates/gates_kernels.cu
    src/kernels/gpu_kernels.cu
    src/kernels/quantum_kernels.cu
)

set(CIRCUIT_SOURCES
    src/circuits/cat_state_circuit.cpp
    src/circuits/gkp_state_circuit.cpp
    src/circuits/jch_simulation_circuit.cpp
    src/circuits/qaoa_circuit.cpp
    src/circuits/qft_circuit.cpp
    src/circuits/shors_circuit.cpp
    src/circuits/state_transfer_circuit.cpp
    src/circuits/vqe_circuit.cpp
    src/run_sc26_bosonic.cpp
)

cuda_add_library(gpu_kernels STATIC ${CUDA_SOURCES})
target_link_libraries(gpu_kernels ${CUDA_LIBRARIES})

# 为每个电路创建一个不包含 main 函数的库文件
foreach(SOURCE ${CIRCUIT_SOURCES})
    if(SOURCE MATCHES "_circuit\.cpp$")
        get_filename_component(LIBRARY_NAME ${SOURCE} NAME_WE)
        set(LIBRARY_NAME "${LIBRARY_NAME}_lib")
        
        # 创建一个不包含 main 函数的库文件
        cuda_add_library(${LIBRARY_NAME} STATIC ${SOURCE} ${COMMON_SOURCES})
        target_compile_definitions(${LIBRARY_NAME} PRIVATE NO_MAIN)
        target_link_libraries(${LIBRARY_NAME} gpu_kernels ${CUDA_LIBRARIES})
        
        # 编译带有 main 函数的可执行文件
        get_filename_component(EXECUTABLE_NAME ${SOURCE} NAME_WE)
        cuda_add_executable(${EXECUTABLE_NAME} ${SOURCE} ${COMMON_SOURCES})
        target_link_libraries(${EXECUTABLE_NAME} gpu_kernels ${CUDA_LIBRARIES})
    endif()
endforeach()

# 编译 run_sc26_bosonic 可执行文件，链接所有电路库
cuda_add_executable(run_sc26_bosonic src/run_sc26_bosonic.cpp ${COMMON_SOURCES})
target_link_libraries(run_sc26_bosonic 
    gpu_kernels 
    cat_state_circuit_lib
    gkp_state_circuit_lib
    jch_simulation_circuit_lib
    qaoa_circuit_lib
    qft_circuit_lib
    shors_circuit_lib
    state_transfer_circuit_lib
    vqe_circuit_lib
    ${CUDA_LIBRARIES}
)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/result)