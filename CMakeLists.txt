cmake_minimum_required(VERSION 3.10)

project(implicit_dist VERSION 0.0.1)
set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
set(CMAKE_CXX_STANDARD 17)

add_compile_definitions(EIGEN_INITIALIZE_MATRICES_BY_NAN)
add_compile_definitions(EIGEN_DONT_VECTORIZE)

# Set variables
set(LIBRARY_NAME   ${PROJECT_NAME})
set(LIBRARY_FOLDER ${PROJECT_NAME})
include(${PROJECT_SOURCE_DIR}/cmake/SetEnv.cmake)

# Eigen
find_package(Eigen3 3.4 REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

#find_package(PoseLib REQUIRED)

# Library sources
add_subdirectory(implicit_dist)

# python bindings
add_subdirectory(pybind)

# Compilation options
if(MSVC)
	target_compile_options(${LIBRARY_NAME} PRIVATE /bigobj /fp:fast)
else()
	target_compile_options(${LIBRARY_NAME} PRIVATE
		-march=native -Wall -Werror -Wno-sign-compare
	 	-Wno-unused-variable -ffast-math -fPIC)
endif()


