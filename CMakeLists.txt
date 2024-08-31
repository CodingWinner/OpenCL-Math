cmake_minimum_required(VERSION 3.30.3)
project("linearalgebra")

set(CMAKE_C_STANDARD 11)

#Executable part
add_executable(${PROJECT_NAME} main.c)
include_directories(${CMAKE_SOURCE_DIR}/OpenCL-Headers)
target_link_libraries(${PROJECT_NAME} OpenCL/libopencl)

add_custom_target(run
    COMMAND ${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)