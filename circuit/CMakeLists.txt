cmake_minimum_required(VERSION 3.10)
project(HybridCVDV_Circuits)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含目录
include_directories(../include)

# 源文件目录
link_directories(../build)

# 找到CUDA库
find_package(CUDA REQUIRED)

# 电路可执行文件
add_executable(cat_state_circuit src/cat_state_circuit.cpp)
target_link_libraries(cat_state_circuit HybridCVDV-Simulator ${CUDA_LIBRARIES})

add_executable(gkp_state_circuit src/gkp_state_circuit.cpp)
target_link_libraries(gkp_state_circuit HybridCVDV-Simulator ${CUDA_LIBRARIES})

add_executable(state_transfer_circuit src/state_transfer_circuit.cpp)
target_link_libraries(state_transfer_circuit HybridCVDV-Simulator ${CUDA_LIBRARIES})

add_executable(jch_simulation_circuit src/jch_simulation_circuit.cpp)
target_link_libraries(jch_simulation_circuit HybridCVDV-Simulator ${CUDA_LIBRARIES})

add_executable(vqe_circuit src/vqe_circuit.cpp)
target_link_libraries(vqe_circuit HybridCVDV-Simulator ${CUDA_LIBRARIES})

add_executable(qaoa_circuit src/qaoa_circuit.cpp)
target_link_libraries(qaoa_circuit HybridCVDV-Simulator ${CUDA_LIBRARIES})

add_executable(shors_circuit src/shors_circuit.cpp)
target_link_libraries(shors_circuit HybridCVDV-Simulator ${CUDA_LIBRARIES})

add_executable(qft_circuit src/qft_circuit.cpp)
target_link_libraries(qft_circuit HybridCVDV-Simulator ${CUDA_LIBRARIES})
