cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

project(pkdMaskCreation)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(pkdMaskCreation code2.cxx)


target_link_libraries(pkdMaskCreation ${ITK_LIBRARIES})