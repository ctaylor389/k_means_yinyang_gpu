
set(lib_name "${PROJECT_NAME}_cuda")
set(exe_name "${PROJECT_NAME}_main")


set(lib_source
    "${CMAKE_CURRENT_SOURCE_DIR}/kmeansCPU.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/kmeansGPU.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/kmeansKernel.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/kmeansUtil.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/kmeansTest.cu"
)

set(main_source "${CMAKE_CURRENT_SOURCE_DIR}/kmeansMain.cu")

add_library(${lib_name} ${lib_source})
add_executable(${exe_name} ${main_source})

target_link_libraries(${exe_name} ${lib_name})