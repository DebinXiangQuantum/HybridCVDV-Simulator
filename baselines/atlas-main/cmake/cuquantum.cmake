# 使用 CMakeLists.txt 中已经设置的变量

# find cuda and custatevec
if(CUSTATEVEC_FOUND)
  list(APPEND QSIM_EXT_LIBRARIES
    ${CUSTATEVEC_LIBRARIES})

  list(APPEND QSIM_INCLUDE_DIRS
    ${CUSTATEVEC_INCLUDE_DIR})
endif()

if(CUSTATEVEC_FOUND)
  message( STATUS "CUSTATEVEC inlcude : ${CUSTATEVEC_INCLUDE_DIR}" )
  message( STATUS "CUSTATEVEC libraries : ${CUSTATEVEC_LIBRARIES}" )
  message("QSIM_INCLUDE_DIRS cuquantum: ${QSIM_INCLUDE_DIRS}")
else()
  message( FATAL_ERROR "CUSTATEVEC package not found -> specify search path via CUQUANTUM_DIR variable")
endif()
