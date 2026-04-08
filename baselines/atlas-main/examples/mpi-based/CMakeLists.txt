cmake_minimum_required(VERSION 3.10)

project(QSIM_Test)

# 原有的 simulate 目标
set(simulate_target simulate)
set(CPU_SRC
  test_sim.cc)

cuda_add_executable(${simulate_target} ${CPU_SRC})
target_include_directories(${simulate_target} PRIVATE ${QSIM_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
message("QSIM_INCLUDE_DIRS: ${QSIM_INCLUDE_DIRS}")
target_link_libraries(${simulate_target} -Wl,--whole-archive quansim -Wl,--no-whole-archive ${QSIM_EXT_LIBRARIES})

# 新的 run_generated_qasm 目标
set(run_qasm_target run_generated_qasm)
set(RUN_QASM_SRC
  run_generated_qasm.cc)

cuda_add_executable(${run_qasm_target} ${RUN_QASM_SRC})
target_include_directories(${run_qasm_target} PRIVATE ${QSIM_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${run_qasm_target} -Wl,--whole-archive quansim -Wl,--no-whole-archive ${QSIM_EXT_LIBRARIES} ${CMAKE_BINARY_DIR}/deps/quartz/libquartz_runtime.so nvidia-ml)

# 新的 test_param 目标
set(test_param_target test_param)
set(TEST_PARAM_SRC
  test_param.cc)

cuda_add_executable(${test_param_target} ${TEST_PARAM_SRC})
target_include_directories(${test_param_target} PRIVATE ${QSIM_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${test_param_target} -Wl,--whole-archive quansim -Wl,--no-whole-archive ${QSIM_EXT_LIBRARIES})

set(BIN_DEST "bin")
install(TARGETS ${simulate_target} ${run_qasm_target} ${test_param_target} DESTINATION ${BIN_DEST})