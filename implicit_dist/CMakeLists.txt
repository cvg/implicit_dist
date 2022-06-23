set(SOURCES
    oned_radial.cc
    cost_matrix.cc    
    pose_refinement.cc
    ordering.cc
    intrinsic.cc
    bundle_adjustment.cc
    utils.cc
    camera_pose.cc
)


# Set HEADERS_PUBLIC variable
set(HEADERS_PUBLIC    
    oned_radial.h
    cost_matrix.h
    cost_functions.h
    pose_refinement.h
    ordering.h
    intrinsic.h
    bundle_adjustment.h
    utils.h    
    camera_pose.h
)

# library configuration
include(${PROJECT_SOURCE_DIR}/cmake/LibraryConfig.cmake)
target_link_libraries(implicit_dist PUBLIC Eigen3::Eigen ${CERES_LIBRARIES})
target_compile_features(implicit_dist PRIVATE cxx_std_17)

