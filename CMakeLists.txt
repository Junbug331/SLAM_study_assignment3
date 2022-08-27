cmake_minimum_required(VERSION 3.15)
project(SLAM-assignment3)

set(TARGET ${CMAKE_PROJECT_NAME})
set(EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(SRC_FILES 
    src/main.cpp)
add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    
include_directories(include)

add_subdirectory(utils)

add_executable(example ${SRC_FILES})

target_link_libraries(example PRIVATE
    utils
)

