cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

project(CystsSegmentation)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(CystsSegmentation 3.2.cxx)


target_link_libraries(CystsSegmentation ${ITK_LIBRARIES})