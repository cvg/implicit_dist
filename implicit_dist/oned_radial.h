#ifndef IMPLICIT_DIST_ONED_RADIAL_H_
#define IMPLICIT_DIST_ONED_RADIAL_H_

#include <Eigen/Dense>
// #include "types.h"
#include "camera_pose.h"
#include <vector>

namespace implicit_dist {

    struct PoseRefinement1DRadialOptions {
        bool weight_residuals = true;
        bool verbose = false;

        PoseRefinement1DRadialOptions clone() const {
            PoseRefinement1DRadialOptions copy = *this;
            return copy;
        }
    };

    void pose_refinement_1D_radial(const std::vector<Eigen::Vector2d> &points2D,
                                const std::vector<Eigen::Vector3d> &points3D,
                                CameraPose *pose, Eigen::Vector2d *pp, 
                                PoseRefinement1DRadialOptions opt);

    void joint_pose_refinement_1D_radial(const std::vector<std::vector<Eigen::Vector2d>> &points2D,
                                const std::vector<std::vector<Eigen::Vector3d>> &points3D,
                                std::vector<CameraPose> *pose, Eigen::Vector2d *pp, 
                                PoseRefinement1DRadialOptions opt);

}

#endif