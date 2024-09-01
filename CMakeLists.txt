cmake_minimum_required(VERSION 3.30.3)
project("linearalgebra")

set(CMAKE_C_STANDARD 11)
set(ENABLED_LANGUAGES "English")

#Executable part
add_executable(${PROJECT_NAME} main.c)
include_directories(${CMAKE_SOURCE_DIR}/OpenCL-Headers)
include_directories(${CMAKE_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE "C:/C and C++/OpenCL Math/libopencl.dll")

add_custom_target(run
    COMMAND ${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)