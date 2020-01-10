cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

project(pkdVolumeComputation)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(pkdVolumeComputation code4.cxx)


target_link_libraries(pkdVolumeComputation ${ITK_LIBRARIES})