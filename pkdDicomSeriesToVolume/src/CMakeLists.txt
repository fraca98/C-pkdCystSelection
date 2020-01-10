cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

project(pkdDicomSeriesToVolume)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(pkdDicomSeriesToVolume code1.cxx)


target_link_libraries(pkdDicomSeriesToVolume ${ITK_LIBRARIES})
