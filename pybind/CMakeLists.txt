# setup PYTHON_EXECUTABLE

find_package (Python COMPONENTS Interpreter Development)
message(STATUS "Python_EXECUTABLE: " ${Python_EXECUTABLE})

add_subdirectory(pybind11)

pybind11_add_module(pyimplicitdist MODULE pyimplicitdist.cpp)

target_link_libraries(pyimplicitdist PUBLIC implicit_dist Eigen3::Eigen ${CERES_LIBRARIES})
target_compile_features(pyimplicitdist PRIVATE cxx_std_17)
