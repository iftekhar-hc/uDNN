cmake_minimum_required(VERSION 3.8)
project(include CXX)

set(CMAKE_CXX_STANDARD 14)


add_subdirectory(extern/googletest/)
# add_subdirectory(extern/pybind11)
# include_directories(/usr/include/python2.7/)

# SIMD

include(GoogleTest)
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
# add_subdirectory(python)
