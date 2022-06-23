#ifndef IMPLICIT_DIST_POSE_REFINEMENT_H_
#define IMPLICIT_DIST_POSE_REFINEMENT_H_

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "camera_pose.h"
#include "cost_matrix.h"
#include "intrinsic.h"
#include <limits>
#include <vector>

namespace implicit_dist {

    struct PoseRefinementOptions {

        // Robust loss functions
        enum LossFunction {
            TRIVIAL,
            HUBER,
            CAUCHY
        };
        
        LossFunction loss_radial = HUBER;
        double loss_scale_radial = 1.0;
        LossFunction loss_dist = HUBER;
        double loss_scale_dist = 1.0;

        // threshold for filtering
        double filter_thres = 10.0;

        bool verbose = false; // whether to print ceres opt

        PoseRefinementOptions clone() const {
            PoseRefinementOptions copy = *this;
            return copy;
        }
    };
    
    // Pose refinement with the implicit distortion modeling
    
    CameraPose pose_refinement(const std::vector<Eigen::Vector2d> &points2D, 
                                const std::vector<Eigen::Vector3d> &points3D,
                                const CostMatrix &cost_matrix, const Eigen::Vector2d &pp,
                                const CameraPose &initial_pose, PoseRefinementOptions refinement_opt);

    std::vector<CameraPose> pose_refinement_multi(const std::vector<std::vector<Eigen::Vector2d>> &points2D, 
                                const std::vector<std::vector<Eigen::Vector3d>> &points3D,
                                const CostMatrix &cost_matrix, const Eigen::Vector2d &pp,
                                const std::vector<CameraPose> &initial_poses, PoseRefinementOptions refinement_opt);

    // Refines camera pose using a fixed non-parametric intrinsic calibration
    CameraPose non_parametric_pose_refinement(const std::vector<Eigen::Vector2d> &points2D, 
                                const std::vector<Eigen::Vector3d> &points3D,
                                const IntrinsicCalib &calib,
                                const CameraPose &initial_pose, PoseRefinementOptions refinement_opt);

    void filter_result_pose_refinement(std::vector<Eigen::Vector2d> &points2D,
                                std::vector<Eigen::Vector3d> &points3D,
                                const CameraPose& pose, const Eigen::Vector2d &pp, 
                                PoseRefinementOptions refinement_opt);

    void filter_result_pose_refinement_multi(std::vector<std::vector<Eigen::Vector2d>> &points2D,
                                std::vector<std::vector<Eigen::Vector3d>> &points3D,
                                const std::vector<CameraPose>& poses, const Eigen::Vector2d &pp, 
                                PoseRefinementOptions refinement_opt);
}

#endif