cmake_minimum_required(VERSION 3.10)
project(UTILS)

set(CMAKE_CXX_STANDARD 17)
set(UTILS_SRC_FILES
    src/utils.cpp)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    message(STATUS "Eigen3 Found - ${Eigen3_DIR}")
endif()

add_library(utils ${UTILS_SRC_FILES})

target_include_directories(utils PUBLIC
    include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(utils PUBLIC
    Eigen3::Eigen
    stdc++fs
    ${OpenCV_LIBS}
)
